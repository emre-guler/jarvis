"""Voice profile encryption module"""
import logging
from pathlib import Path
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import os
import numpy as np
import json
from typing import Optional, Dict, Union, Any

logger = logging.getLogger(__name__)

class ProfileEncryption:
    def __init__(self, key_file: Optional[Path] = None):
        """Initialize the profile encryption system
        
        Args:
            key_file: Optional path to encryption key file. If not provided,
                     uses default location in data/security/encryption.key
        """
        self.key_file = key_file or Path(__file__).parent.parent.parent.parent / "data" / "security" / "encryption.key"
        self.key_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize or load encryption key
        self.key = self._get_encryption_key()
        self.fernet = Fernet(self.key)
        
    def _get_encryption_key(self) -> bytes:
        """Get or generate encryption key"""
        try:
            if self.key_file.exists():
                with open(self.key_file, 'rb') as f:
                    return f.read()
            
            # Generate new key
            salt = os.urandom(16)
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000
            )
            key = base64.urlsafe_b64encode(kdf.derive(os.urandom(32)))
            
            # Save key
            with open(self.key_file, 'wb') as f:
                f.write(key)
            
            return key
            
        except Exception as e:
            logger.error(f"Error getting encryption key: {e}")
            raise
            
    def encrypt_profile(self, data: Union[bytes, Dict[str, Any]]) -> bytes:
        """Encrypt voice profile data
        
        Args:
            data: Voice profile data (bytes or dictionary)
            
        Returns:
            bytes: Encrypted data
        """
        try:
            # Convert dictionary to JSON string if needed
            if isinstance(data, dict):
                data = json.dumps(data).encode('utf-8')
            elif not isinstance(data, bytes):
                raise ValueError("Data must be either bytes or dictionary")
            
            # Encrypt
            encrypted_data = self.fernet.encrypt(data)
            
            return encrypted_data
            
        except Exception as e:
            logger.error(f"Error encrypting data: {e}")
            raise
            
    def decrypt_profile(self, encrypted_data: bytes) -> Union[bytes, Dict[str, Any]]:
        """Decrypt voice profile data
        
        Args:
            encrypted_data: Encrypted profile data
            
        Returns:
            Union[bytes, Dict[str, Any]]: Decrypted data
        """
        try:
            # Decrypt data
            decrypted_data = self.fernet.decrypt(encrypted_data)
            
            # Try to parse as JSON
            try:
                # Only try JSON parsing if it starts with { or [
                if decrypted_data[0] in (b'{'[0], b'['[0]):
                    return json.loads(decrypted_data.decode('utf-8'))
            except (json.JSONDecodeError, UnicodeDecodeError, IndexError):
                pass
                
            # Return as raw bytes if not JSON
            return decrypted_data
            
        except Exception as e:
            logger.error(f"Error decrypting data: {e}")
            raise
            
    def encrypt_data(self, data: np.ndarray) -> bytes:
        """Encrypt numpy array data
        
        Args:
            data: Numpy array to encrypt
            
        Returns:
            bytes: Encrypted data
        """
        try:
            # Convert numpy array to bytes
            data_bytes = data.tobytes()
            
            # Encrypt
            encrypted_data = self.fernet.encrypt(data_bytes)
            
            return encrypted_data
            
        except Exception as e:
            logger.error(f"Error encrypting data: {e}")
            raise
            
    def decrypt_data(self, encrypted_data: bytes) -> np.ndarray:
        """Decrypt numpy array data
        
        Args:
            encrypted_data: Encrypted data
            
        Returns:
            np.ndarray: Decrypted numpy array
        """
        try:
            # Decrypt data
            decrypted_data = self.fernet.decrypt(encrypted_data)
            
            # Convert back to numpy array
            array_data = np.frombuffer(decrypted_data)
            
            return array_data
            
        except Exception as e:
            logger.error(f"Error decrypting data: {e}")
            raise 