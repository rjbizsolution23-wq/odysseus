"""
uncensored_media_server.py

MCP server exposing an Uncensored/Unfiltered Media Engine with dynamic, on-the-fly LoRA merging.
Supports Replicate and Fal.ai pipelines.
"""

import asyncio
import base64
import os
import sys
import uuid
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Setup logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("uncensored_media_server")

server = Server("uncensored_media")

@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="generate_uncensored_image",
            description=(
                "Generate a high-fidelity image using specialized unfiltered/uncensored models "
                "(like FLUX.1 Dev/Schnell or SDXL) with optional on-the-fly dynamic LoRA merging. "
                "Can fetch LoRAs from Hugging Face or Civitai on-the-fly."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "prompt": {"type": "string", "description": "Highly detailed image description. Be descriptive."},
                    "negative_prompt": {"type": "string", "description": "Negative prompts (if using SDXL or models that support it)"},
                    "loras": {
                        "type": "array",
                        "description": "List of dynamic LoRAs to merge on-the-fly.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "url": {"type": "string", "description": "Hugging Face or Civitai .safetensors direct download URL"},
                                "weight": {"type": "number", "description": "LoRA weight, e.g. 0.8, -1.0 to 2.0 (default 1.0)"},
                                "trigger_word": {"type": "string", "description": "Optional keyword/trigger word for the LoRA"}
                            },
                            "required": ["url"]
                        }
                    },
                    "aspect_ratio": {"type": "string", "enum": ["1:1", "16:9", "9:16", "3:2", "2:3", "4:5"], "description": "Image aspect ratio (default 1:1)"},
                    "num_inference_steps": {"type": "integer", "description": "Steps for generation (default 28-35 for FLUX)"},
                    "guidance_scale": {"type": "number", "description": "Guidance scale (default 3.5)"},
                    "seed": {"type": "integer", "description": "Generation seed for reproducibility"}
                },
                "required": ["prompt"],
            },
        ),
        Tool(
            name="generate_uncensored_video",
            description="Generate a high-fidelity video clip from text (or animate an existing image) using advanced open-weights video models (CogVideoX, HunyuanVideo, LTX-Video, Kling).",
            inputSchema={
                "type": "object",
                "properties": {
                    "prompt": {"type": "string", "description": "Highly detailed video description prompt."},
                    "image_url": {"type": "string", "description": "Optional source image URL or local path to animate/use as reference for Image-to-Video generation."},
                    "negative_prompt": {"type": "string", "description": "Optional elements to avoid in the video."},
                    "aspect_ratio": {"type": "string", "enum": ["16:9", "9:16", "1:1"], "description": "Video aspect ratio (default 16:9)"},
                    "duration": {"type": "string", "enum": ["5s", "10s"], "description": "Video duration in seconds (default 5s)"},
                    "quality": {"type": "string", "enum": ["high", "medium", "low"], "description": "Quality preset"}
                },
                "required": ["prompt"],
            }
        ),
        Tool(
            name="generate_premium_music_video",
            description="Generate a fully synchronized premium music video syncing cinematic open-source video and open-source audio tracks using Google TPU/GPU resources.",
            inputSchema={
                "type": "object",
                "properties": {
                    "music_prompt": {"type": "string", "description": "Description of original background track/music theme."},
                    "video_prompt": {"type": "string", "description": "Cinematic visual details to render for the video clip."}
                },
                "required": ["music_prompt", "video_prompt"]
            }
        ),
        Tool(
            name="nvidia_nim_inference",
            description="Execute ultra-low-latency serverless inference or processing on a premium model via the NVIDIA Developer Platform NIM endpoint.",
            inputSchema={
                "type": "object",
                "properties": {
                    "model": {"type": "string", "description": "Model ID to query (e.g. nvidia/nemotron-3-ultra-550b-a55b)"},
                    "messages": {
                        "type": "array",
                        "description": "List of message objects (role & content)",
                        "items": {
                            "type": "object",
                            "properties": {
                               "role": {"type": "string"},
                               "content": {"type": "string"}
                            },
                            "required": ["role", "content"]
                        }
                    },
                    "temperature": {"type": "number", "description": "Sampling temperature (default 0.7)"},
                    "max_tokens": {"type": "integer", "description": "Max tokens to generate"}
                },
                "required": ["model", "messages"]
            }
        )
    ]

@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    logger.info(f"Uncensored Media Tool call: {name} with args: {arguments}")

    if name == "generate_uncensored_image":
        return await handle_generate_image(arguments)
    elif name == "generate_uncensored_video":
        return await handle_generate_video(arguments)
    elif name == "generate_premium_music_video":
        return await handle_generate_music_video(arguments)
    elif name == "nvidia_nim_inference":
        return await handle_nvidia_inference(arguments)
    else:
        return [TextContent(type="text", text=f"Unknown tool: {name}")]

async def handle_generate_image(args: dict) -> list[TextContent]:
    prompt = args.get("prompt", "")
    negative_prompt = args.get("negative_prompt", "")
    loras = args.get("loras", [])
    aspect_ratio = args.get("aspect_ratio", "1:1")
    steps = args.get("num_inference_steps", 28)
    guidance = args.get("guidance_scale", 3.5)
    seed = args.get("seed", None)

    # Load keys
    from src.settings import get_setting
    replicate_token = os.getenv("REPLICATE_API_TOKEN") or get_setting("REPLICATE_API_TOKEN")
    fal_key = os.getenv("FAL_API_KEY") or get_setting("FAL_API_KEY")

    if not replicate_token and not fal_key:
        return [TextContent(type="text", text="Error: Neither REPLICATE_API_TOKEN nor FAL_API_KEY is configured in the environment.")]

    # Append any trigger words from Loras to the prompt
    for lora in loras:
        trig = lora.get("trigger_word")
        if trig and trig.lower() not in prompt.lower():
            prompt = f"{prompt}, {trig}"

    # Use Fal.ai if key is present, otherwise use Replicate
    if fal_key:
        try:
            import httpx
            logger.info("Using FAL.ai for uncensored image generation.")
            # Map parameters for Fal flux-lora endpoint
            fal_url = "https://queue.fal.run/fal-ai/flux-lora"
            headers = {
                "Authorization": f"Key {fal_key}",
                "Content-Type": "application/json"
            }
            
            # Format loras for Fal
            # Fal expects: [{ "path": "url", "scale": weight }]
            fal_loras = []
            for lora in loras:
                fal_loras.append({
                    "path": lora["url"],
                    "scale": lora.get("weight", 1.0)
                })

            payload = {
                "prompt": prompt,
                "image_size": aspect_ratio_to_size(aspect_ratio),
                "num_inference_steps": steps,
                "guidance_scale": guidance,
                "enable_safety_checker": False, # Uncensored/unfiltered
            }
            if fal_loras:
                payload["loras"] = fal_loras
            if seed is not None:
                payload["seed"] = seed

            async with httpx.AsyncClient(timeout=300.0) as client:
                resp = await client.post(fal_url, json=payload, headers=headers)
                if resp.status_code != 200:
                    return [TextContent(type="text", text=f"Fal.ai Error ({resp.status_code}): {resp.text}")]
                
                res_data = resp.json()
                # If queued, poll for results
                request_id = res_data.get("request_id")
                if request_id:
                    status_url = f"https://queue.fal.run/fal-ai/flux-lora/requests/{request_id}"
                    for _ in range(60): # Poll up to 5 mins
                        await asyncio.sleep(5)
                        status_resp = await client.get(status_url, headers=headers)
                        if status_resp.status_code == 200:
                            status_data = status_resp.json()
                            if status_data.get("status") == "COMPLETED":
                                res_data = status_data.get("logs") or status_data.get("response") or status_data
                                break
                            elif status_data.get("status") == "FAILED":
                                return [TextContent(type="text", text=f"Fal.ai Queue Job Failed: {status_data}")]
                
                images = res_data.get("images", [])
                if not images:
                    # check if response is nested
                    images = res_data.get("response", {}).get("images", [])

                if images:
                    image_url = images[0].get("url")
                    # Save to local gallery
                    res_url = await download_and_save_to_gallery(image_url, prompt, "fal-ai/flux-lora", aspect_ratio)
                    final_url = res_url if (res_url.startswith("http://") or res_url.startswith("https://")) else f"/api/generated-image/{res_url}"
                    return [TextContent(type="text", text=f"Successfully generated uncensored image!\nImage URL: {final_url}\nPrompt: {prompt}")]

        except Exception as e:
            logger.error(f"Fal.ai generation failed: {e}. Falling back to Replicate.")

    # Replicate fallback / direct path
    if replicate_token:
        try:
            import httpx
            logger.info("Using Replicate for uncensored image generation.")
            # We use flux-dev-multi-lora or similar customizable model
            replicate_url = "https://api.replicate.com/v1/predictions"
            headers = {
                "Authorization": f"Token {replicate_token}",
                "Content-Type": "application/json"
            }

            # Map parameters for lucataco/flux-dev-multi-lora
            # It expects comma separated loras and weights
            loras_str = ",".join([l["url"] for l in loras])
            weights_str = ",".join([str(l.get("weight", 1.0)) for l in loras])

            payload = {
                "version": "14f2e519c23577d612e61a297a73fcb5c777494517b1897f223f6e1f0e42d765", # lucataco/flux-dev-multi-lora
                "input": {
                    "prompt": prompt,
                    "aspect_ratio": aspect_ratio,
                    "num_inference_steps": steps,
                    "guidance_scale": guidance,
                    "disable_safety_checker": True # Uncensored
                }
            }
            if loras_str:
                payload["input"]["loras"] = loras_str
                payload["input"]["lora_weights"] = weights_str
            if seed is not None:
                payload["input"]["seed"] = seed

            async with httpx.AsyncClient(timeout=300.0) as client:
                resp = await client.post(replicate_url, json=payload, headers=headers)
                if resp.status_code != 201:
                    return [TextContent(type="text", text=f"Replicate API Error ({resp.status_code}): {resp.text}")]
                
                pred = resp.json()
                poll_url = pred["urls"]["get"]
                
                # Poll Replicate prediction
                for _ in range(60):
                    await asyncio.sleep(5)
                    poll_resp = await client.get(poll_url, headers=headers)
                    if poll_resp.status_code == 200:
                        pred_data = poll_resp.json()
                        status = pred_data.get("status")
                        if status == "succeeded":
                            output = pred_data.get("output")
                            if isinstance(output, list) and output:
                                image_url = output[0]
                                res_url = await download_and_save_to_gallery(image_url, prompt, "lucataco/flux-dev-multi-lora", aspect_ratio)
                                final_url = res_url if (res_url.startswith("http://") or res_url.startswith("https://")) else f"/api/generated-image/{res_url}"
                                return [TextContent(type="text", text=f"Successfully generated uncensored image via Replicate!\nImage URL: {final_url}\nPrompt: {prompt}")]
                            elif isinstance(output, str):
                                res_url = await download_and_save_to_gallery(output, prompt, "lucataco/flux-dev-multi-lora", aspect_ratio)
                                final_url = res_url if (res_url.startswith("http://") or res_url.startswith("https://")) else f"/api/generated-image/{res_url}"
                                return [TextContent(type="text", text=f"Successfully generated uncensored image via Replicate!\nImage URL: {final_url}\nPrompt: {prompt}")]
                        elif status in ("failed", "canceled"):
                            return [TextContent(type="text", text=f"Replicate generation failed: {pred_data.get('error') or 'Canceled'}")]

        except Exception as e:
            return [TextContent(type="text", text=f"Error running Replicate prediction: {e}")]

    return [TextContent(type="text", text="Error: Model execution failed on all configured endpoints.")]

async def ensure_public_url(url_or_path: str) -> str:
    if not url_or_path:
        return ""
    if url_or_path.startswith("http://") or url_or_path.startswith("https://"):
        return url_or_path

    # Try resolving local path
    path = Path(url_or_path)
    
    # If it is a relative API path
    if url_or_path.startswith("/api/generated-image/"):
        filename = url_or_path.replace("/api/generated-image/", "")
        path = Path("data/generated_images") / filename
    elif url_or_path.startswith("/api/generated-video/"):
        filename = url_or_path.replace("/api/generated-video/", "")
        path = Path("data/generated_videos") / filename

    if not path.exists():
        # Maybe it's a relative path in the app
        potential_paths = [
            Path(url_or_path),
            Path("data/generated_images") / url_or_path,
            Path("data/generated_videos") / url_or_path,
        ]
        for p in potential_paths:
            if p.exists():
                path = p
                break

    if path.exists() and path.is_file():
        # Try R2 Upload first
        try:
            from src.cloudflare_r2 import CloudflareR2Adapter
            r2 = CloudflareR2Adapter()
            if r2.is_active:
                # Determine mime type
                ext = path.suffix.lower()
                mime_type = "image/png" if ext in (".png", ".jpg", ".jpeg") else "video/mp4"
                r2_key = f"temp/{path.name}"
                r2_url = r2.upload_file(path, r2_key, mime_type)
                if r2_url:
                    logger.info(f"Uploaded local asset {path} to R2: {r2_url}")
                    return r2_url
        except Exception as e:
            logger.error(f"Failed to upload local asset to R2: {e}")

        # Fallback to data URI if it's an image and not too big
        try:
            ext = path.suffix.lower()
            if ext in (".png", ".jpg", ".jpeg", ".webp"):
                mime_type = "image/webp" if ext == ".webp" else ("image/jpeg" if ext in (".jpg", ".jpeg") else "image/png")
                with open(path, "rb") as f:
                    encoded = base64.b64encode(f.read()).decode("utf-8")
                return f"data:{mime_type};base64,{encoded}"
        except Exception as e:
            logger.error(f"Failed to create data URI: {e}")

    return url_or_path

async def handle_generate_video(args: dict) -> list[TextContent]:
    prompt = args.get("prompt", "")
    image_url = args.get("image_url", "")
    aspect_ratio = args.get("aspect_ratio", "16:9")
    duration = args.get("duration", "5s")
    
    from src.settings import get_setting
    replicate_token = os.getenv("REPLICATE_API_TOKEN") or get_setting("REPLICATE_API_TOKEN")
    
    if not replicate_token:
        return [TextContent(type="text", text="Error: REPLICATE_API_TOKEN is not configured for video generation.")]

    try:
        import httpx
        replicate_url = "https://api.replicate.com/v1/predictions"
        headers = {
            "Authorization": f"Token {replicate_token}",
            "Content-Type": "application/json"
        }

        if image_url:
            resolved_image_url = await ensure_public_url(image_url)
            logger.info(f"Using LTX-Video for Image-to-Video generation using reference image: {resolved_image_url}")
            # Use lightricks/ltx-video or lucataco/ltx-video which takes text + image
            payload = {
                "version": "7823f6634c8c7f9996c561b6cb1b181db8fa087bc78fa1b1e95e7b27be361419", # ltx-video
                "input": {
                    "prompt": prompt,
                    "input_image": resolved_image_url,
                    "aspect_ratio": aspect_ratio,
                    "video_length": "5s" if duration == "5s" else "10s",
                    "disable_safety_checker": True
                }
            }
        else:
            logger.info("Using HunyuanVideo for Text-to-Video generation.")
            payload = {
                "version": "8550b047e1195fbc2bf9e55198a00bde48bb99478f595e7b27be36141977a450", # hunyuan-video
                "input": {
                    "prompt": prompt,
                    "aspect_ratio": aspect_ratio,
                    "video_length": "5s" if duration == "5s" else "10s",
                    "disable_safety_checker": True
                }
            }

        async with httpx.AsyncClient(timeout=300.0) as client:
            resp = await client.post(replicate_url, json=payload, headers=headers)
            if resp.status_code != 201:
                return [TextContent(type="text", text=f"Replicate Video Error ({resp.status_code}): {resp.text}")]
            
            pred = resp.json()
            poll_url = pred["urls"]["get"]
            
            # Poll Replicate prediction
            for _ in range(120): # Video takes longer
                await asyncio.sleep(5)
                poll_resp = await client.get(poll_url, headers=headers)
                if poll_resp.status_code == 200:
                    pred_data = poll_resp.json()
                    status = pred_data.get("status")
                    if status == "succeeded":
                        output = pred_data.get("output")
                        video_url = output if isinstance(output, str) else (output[0] if isinstance(output, list) else None)
                        if video_url:
                            # Save to local data/generated_videos
                             res_url = await download_and_save_video(video_url)
                             final_url = res_url if (res_url.startswith("http://") or res_url.startswith("https://")) else f"/api/generated-video/{res_url}"
                             source_type = "Image-to-Video" if image_url else "Text-to-Video"
                             return [TextContent(type="text", text=f"Successfully generated uncensored {source_type} video!\nVideo File: {final_url}\nPrompt: {prompt}")]
                    elif status in ("failed", "canceled"):
                        return [TextContent(type="text", text=f"Replicate video generation failed: {pred_data.get('error') or 'Canceled'}")]

    except Exception as e:
        return [TextContent(type="text", text=f"Video generation error: {e}")]

    return [TextContent(type="text", text="Error: Video execution failed.")]

async def handle_generate_music_video(args: dict) -> list[TextContent]:
    music_prompt = args.get("music_prompt", "")
    video_prompt = args.get("video_prompt", "")

    if not music_prompt or not video_prompt:
        return [TextContent(type="text", text="Error: Both music_prompt and video_prompt are required.")]

    try:
        from src.gcp_media_orchestrator import GCPMediaOrchestrator
        orch = GCPMediaOrchestrator()
        result = await orch.generate_music_video(music_prompt=music_prompt, video_prompt=video_prompt)
        
        if result.get("success"):
            return [TextContent(type="text", text=f"Successfully generated dynamic open-source Music Video!\n\nVideo URL: {result['music_video_url']}\nAudio URL: {result['audio_url']}\nSource: {result['source']}\n\nGenerated with Google GPU/TPUs and uploaded to Cloudflare R2.")]
        else:
            return [TextContent(type="text", text=f"Failed to generate music video: {result.get('error')}")]
    except Exception as e:
        return [TextContent(type="text", text=f"Error generating music video: {e}")]

async def handle_nvidia_inference(args: dict) -> list[TextContent]:
    model = args.get("model", "nvidia/nemotron-3-ultra-550b-a55b")
    messages = args.get("messages", [])
    temp = args.get("temperature", 0.7)
    max_tokens = args.get("max_tokens", 1024)

    nv_key = os.getenv("NVIDIA_API_KEY")
    if not nv_key:
        return [TextContent(type="text", text="Error: NVIDIA_API_KEY is not configured in the environment.")]

    try:
        import httpx
        url = "https://integrate.api.nvidia.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {nv_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temp,
            "max_tokens": max_tokens,
            "stream": False
        }

        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(url, json=payload, headers=headers)
            if resp.status_code != 200:
                return [TextContent(type="text", text=f"NVIDIA API Error ({resp.status_code}): {resp.text}")]
            
            res = resp.json()
            content = res.get("choices", [{}])[0].get("message", {}).get("content", "")
            return [TextContent(type="text", text=content)]

    except Exception as e:
        return [TextContent(type="text", text=f"NVIDIA NIM error: {e}")]

def aspect_ratio_to_size(ratio: str) -> str:
    mapping = {
        "1:1": "square_hd",
        "16:9": "landscape_hd",
        "9:16": "portrait_hd",
        "3:2": "landscape_hd",
        "2:3": "portrait_hd"
    }
    return mapping.get(ratio, "square_hd")

async def download_and_save_to_gallery(url: str, prompt: str, model_id: str, aspect_ratio: str) -> str:
    import httpx
    img_dir = Path("data/generated_images")
    img_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{uuid.uuid4().hex[:12]}.png"
    img_path = img_dir / filename

    async with httpx.AsyncClient() as client:
        resp = await client.get(url)
        if resp.status_code == 200:
            img_path.write_bytes(resp.content)

    # Save to gallery database
    try:
        from src.database import SessionLocal, GalleryImage
        db = SessionLocal()
        db.add(GalleryImage(
            id=str(uuid.uuid4()),
            filename=filename,
            prompt=prompt,
            model=model_id,
            size=aspect_ratio,
            quality="high",
        ))
        db.commit()
        db.close()
    except Exception as e:
        logger.error(f"Failed saving to gallery: {e}")

    # Cloudflare R2 Upload
    try:
        from src.cloudflare_r2 import CloudflareR2Adapter
        r2 = CloudflareR2Adapter()
        if r2.is_active:
            r2_url = r2.upload_file(img_path, f"images/{filename}", "image/png")
            if r2_url:
                return r2_url
    except Exception as e:
        logger.error(f"R2 image upload failed: {e}")

    return filename

async def download_and_save_video(url: str) -> str:
    import httpx
    vid_dir = Path("data/generated_videos")
    vid_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{uuid.uuid4().hex[:12]}.mp4"
    vid_path = vid_dir / filename

    async with httpx.AsyncClient() as client:
        resp = await client.get(url)
        if resp.status_code == 200:
            vid_path.write_bytes(resp.content)

    # Cloudflare R2 Upload
    try:
        from src.cloudflare_r2 import CloudflareR2Adapter
        r2 = CloudflareR2Adapter()
        if r2.is_active:
            r2_url = r2.upload_file(vid_path, f"videos/{filename}", "video/mp4")
            if r2_url:
                return r2_url
    except Exception as e:
        logger.error(f"R2 video upload failed: {e}")

    return filename

async def run():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())

if __name__ == "__main__":
    asyncio.run(run())
