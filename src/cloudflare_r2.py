# src/cloudflare_r2.py
import logging
import os
import mimetypes
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

class CloudflareR2Adapter:
    """Enterprise-grade adapter for Cloudflare R2 Object Storage.
    
    Provides secure, high-performance file upload, streaming, and URL generation
    leveraging Cloudflare's S3-compatible API.
    """
    def __init__(self):
        self.account_id = os.getenv("CF_ACCOUNT_ID") or os.getenv("CLOUDFLARE_ACCOUNT_ID")
        self.access_key_id = os.getenv("CF_R2_ACCESS_KEY_ID") or os.getenv("CLOUDFLARE_R2_ACCESS_KEY_ID") or os.getenv("CLOUDFLARE_R2_TOKEN")
        self.secret_access_key = os.getenv("CF_R2_SECRET_ACCESS_KEY") or os.getenv("CLOUDFLARE_R2_SECRET_ACCESS_KEY")
        self.bucket_name = os.getenv("CF_R2_BUCKET_NAME") or os.getenv("CLOUDFLARE_R2_BUCKET_NAME") or "jukeyman-assets"
        self.public_domain = os.getenv("CF_R2_PUBLIC_DOMAIN") or os.getenv("CLOUDFLARE_IMAGE_DELIVERY_URL")

        self._s3_client = None
        self._initialized = False

        if all([self.account_id, self.access_key_id, self.secret_access_key]):
            try:
                import boto3
                endpoint_url = f"https://{self.account_id}.r2.cloudflarestorage.com"
                self._s3_client = boto3.client(
                    service_name="s3",
                    endpoint_url=endpoint_url,
                    aws_access_key_id=self.access_key_id,
                    aws_secret_access_key=self.secret_access_key,
                    region_name="auto" # Cloudflare R2 uses 'auto' region
                )
                self._initialized = True
                logger.info(f"Cloudflare R2 Adapter successfully initialized on bucket '{self.bucket_name}'.")
            except ImportError:
                logger.warning("boto3 package not installed. Cloudflare R2 uploads will fall back to local storage. Run 'pip install boto3' to enable.")
            except Exception as e:
                logger.error(f"Failed to initialize Cloudflare R2 Client: {e}")
        else:
            logger.info("Cloudflare R2 credentials not fully specified in environment. Defaulting to local storage.")

    @property
    def is_active(self) -> bool:
        return self._initialized and self._s3_client is not None

    def upload_file(self, local_path: str | Path, r2_key: str, content_type: Optional[str] = None) -> Optional[str]:
        """Upload a local file to Cloudflare R2 bucket.
        
        Args:
            local_path: Path to the local file to upload.
            r2_key: Destination path/name inside the R2 bucket.
            content_type: Optional content-type header. Auto-detected if not specified.
            
        Returns:
            The public URL of the uploaded asset, or None if the upload failed or R2 is disabled.
        """
        if not self.is_active:
            return None

        local_path = Path(local_path)
        if not local_path.exists():
            logger.error(f"Local file not found for R2 upload: {local_path}")
            return None

        if not content_type:
            content_type, _ = mimetypes.guess_type(local_path)
            if not content_type:
                content_type = "application/octet-stream"

        try:
            logger.info(f"Uploading {local_path} to R2 bucket '{self.bucket_name}' key '{r2_key}' with Content-Type '{content_type}'...")
            extra_args = {"ContentType": content_type}
            self._s3_client.upload_file(
                Filename=str(local_path),
                Bucket=self.bucket_name,
                Key=r2_key,
                ExtraArgs=extra_args
            )
            
            # Construct the public URL
            if self.public_domain:
                public_url = f"{self.public_domain.rstrip('/')}/{r2_key}"
            else:
                # Fallback to direct R2 developer public URL / authenticated path if no custom domain
                public_url = f"https://{self.account_id}.r2.cloudflarestorage.com/{self.bucket_name}/{r2_key}"
                
            logger.info(f"File uploaded successfully. Public URL: {public_url}")
            return public_url
        except Exception as e:
            logger.error(f"R2 upload failed: {e}", exc_info=True)
            return None

    def upload_bytes(self, data: bytes, r2_key: str, content_type: str = "application/octet-stream") -> Optional[str]:
        """Upload raw bytes directly to Cloudflare R2 bucket.
        
        Args:
            data: Raw bytes to upload.
            r2_key: Destination path/name inside the R2 bucket.
            content_type: Content type of the data.
            
        Returns:
            The public URL of the uploaded asset, or None if the upload failed or R2 is disabled.
        """
        if not self.is_active:
            return None

        try:
            logger.info(f"Uploading bytes to R2 bucket '{self.bucket_name}' key '{r2_key}'...")
            self._s3_client.put_object(
                Body=data,
                Bucket=self.bucket_name,
                Key=r2_key,
                ContentType=content_type
            )
            
            if self.public_domain:
                public_url = f"{self.public_domain.rstrip('/')}/{r2_key}"
            else:
                public_url = f"https://{self.account_id}.r2.cloudflarestorage.com/{self.bucket_name}/{r2_key}"
                
            logger.info(f"Bytes uploaded successfully. Public URL: {public_url}")
            return public_url
        except Exception as e:
            logger.error(f"R2 bytes upload failed: {e}", exc_info=True)
            return None
