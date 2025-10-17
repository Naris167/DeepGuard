import os
import base64
from datetime import datetime
from pathlib import Path
from typing import Optional, Union
import requests
from dotenv import load_dotenv


class LineNotifier:
    """
    Flexible LINE notification class that only sends fields you provide.
    """
    
    def __init__(self, env_path: str = "./scripts/.env"):
        """
        Initialize LINE Notifier with credentials from .env file.
        
        Args:
            env_path: Path to .env file (default: ./scripts/.env)
        """
        load_dotenv(dotenv_path=env_path)
        
        self.access_token = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
        self.default_group_id = os.getenv("LINE_GROUP_ID")
        self.api_url = os.getenv("LINE_API_URL", "https://api.line.me/v2/bot/message/push")
        
        if not self.access_token:
            raise ValueError("LINE_CHANNEL_ACCESS_TOKEN not found in .env file")
        if not self.default_group_id:
            raise ValueError("LINE_GROUP_ID not found in .env file")
    
    def send_text(self, message: str, group_id: Optional[str] = None) -> dict:
        """
        Send a simple text message to LINE group.
        
        Args:
            message: Text message to send
            group_id: Target group ID (uses default if not provided)
            
        Returns:
            Response dict with status and details
        """
        target_group = group_id or self.default_group_id
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.access_token}"
        }
        
        payload = {
            "to": target_group,
            "messages": [{"type": "text", "text": message}]
        }
        
        try:
            response = requests.post(self.api_url, headers=headers, json=payload, timeout=10)
            response.raise_for_status()
            return {
                "status": "success",
                "message": "Message sent successfully",
                "group_id": target_group
            }
        except requests.exceptions.RequestException as e:
            return {
                "status": "error",
                "message": f"Failed to send message: {str(e)}",
                "group_id": target_group
            }
    
    def send_image(
        self, 
        image_url: str, 
        preview_url: Optional[str] = None,
        group_id: Optional[str] = None
    ) -> dict:
        """
        Send an image message to LINE group.
        
        Args:
            image_url: Public HTTPS URL of the full image
            preview_url: Public HTTPS URL of preview image (optional)
            group_id: Target group ID (uses default if not provided)
            
        Returns:
            Response dict with status and details
        """
        target_group = group_id or self.default_group_id
        preview_url = preview_url or image_url
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.access_token}"
        }
        
        payload = {
            "to": target_group,
            "messages": [
                {
                    "type": "image",
                    "originalContentUrl": image_url,
                    "previewImageUrl": preview_url
                }
            ]
        }
        
        try:
            response = requests.post(self.api_url, headers=headers, json=payload, timeout=10)
            response.raise_for_status()
            return {
                "status": "success",
                "message": "Image sent successfully",
                "group_id": target_group
            }
        except requests.exceptions.RequestException as e:
            return {
                "status": "error",
                "message": f"Failed to send image: {str(e)}",
                "group_id": target_group
            }
    
    def send_fall_detection_alert(
        self,
        camera_id: Optional[str] = None,
        location: Optional[str] = None,
        timestamp: Optional[datetime] = None,
        message: Optional[str] = None,
        image_url: Optional[str] = None,
        group_id: Optional[str] = None
    ) -> dict:
        """
        Send a flexible fall detection alert. Only includes fields you provide.
        Always includes header: "⚠️ เตือน! ตรวจพบคนล้ม"
        
        Args:
            camera_id: Camera identifier (optional)
            location: Location description (optional)
            timestamp: Time of the event (optional)
            message: Additional custom message (optional)
            image_url: Optional HTTPS URL of the incident image
            group_id: Target group ID (uses default if not provided)
            
        Returns:
            Response dict with status and details
        """
        target_group = group_id or self.default_group_id
        
        # Always start with the header
        alert_text = "⚠️ เตือน! ตรวจพบคนล้ม"
        
        # Build message parts conditionally
        message_parts = []
        
        if camera_id is not None:
            message_parts.append(f"📹 กล้อง: {camera_id}")
        
        if location is not None:
            message_parts.append(f"📍 สถานที่: {location}")
        
        if timestamp is not None:
            formatted_time = timestamp.strftime("%Y-%m-%d %H:%M:%S")
            message_parts.append(f"🕐 เวลา: {formatted_time}")
        
        # Add the parts to alert_text if any exist
        if message_parts:
            alert_text += "\n\n" + "\n".join(message_parts)
        
        # Add custom message if provided
        if message is not None:
            alert_text += "\n\n" + message
        
        # Prepare headers
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.access_token}"
        }
        
        # Build messages list
        messages = [{"type": "text", "text": alert_text}]
        
        # Add image if provided
        if image_url:
            messages.append({
                "type": "image",
                "originalContentUrl": image_url,
                "previewImageUrl": image_url
            })
        
        payload = {
            "to": target_group,
            "messages": messages
        }
        
        try:
            response = requests.post(self.api_url, headers=headers, json=payload, timeout=10)
            response.raise_for_status()
            return {
                "status": "success",
                "message": "Fall detection alert sent successfully",
                "group_id": target_group
            }
        except requests.exceptions.RequestException as e:
            return {
                "status": "error",
                "message": f"Failed to send alert: {str(e)}",
                "group_id": target_group
            }