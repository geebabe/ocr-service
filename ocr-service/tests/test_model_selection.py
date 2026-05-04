import sys
import os

# Ensure the local project `app` package takes priority over any system-installed `app` packages
# (e.g., a Flask `app` from Anaconda environment)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
import unittest
from unittest.mock import patch, AsyncMock
from app.services.vlm_factory import vlm_factory
from app.core.config import settings

class TestVLMFactory(unittest.IsolatedAsyncioTestCase):
    
    @patch("app.services.vlm_factory.call_vlm_inference")
    async def test_model_routing_qwen(self, mock_inference):
        mock_inference.return_value = '{"invoice_number": "123"}'
        
        result = await vlm_factory.get_inference("qwen3vl", "dummy_base64")
        
        # Verify it called the correct URL and model name from settings
        mock_inference.assert_called_once_with(
            "dummy_base64", 
            settings.VLLM_URL, 
            settings.VLLM_MODEL
        )
        self.assertEqual(result, '{"invoice_number": "123"}')

    @patch("app.services.vlm_factory.call_vlm_inference")
    async def test_model_routing_vintern(self, mock_inference):
        mock_inference.return_value = '{"invoice_number": "456"}'
        
        result = await vlm_factory.get_inference("vintern", "dummy_base64")
        
        # Verify it called the Vintern URL
        mock_inference.assert_called_once_with(
            "dummy_base64", 
            settings.VINTERN_URL, 
            "Vintern-1B-v2"
        )

    @patch("app.services.vlm_factory.call_vlm_inference")
    async def test_fallback_logic(self, mock_inference):
        mock_inference.return_value = "fallback"
        
        # Call with unknown model
        result = await vlm_factory.get_inference("unknown_model", "dummy_base64")
        
        # Should fallback to default (qwen3vl)
        mock_inference.assert_called_once_with(
            "dummy_base64", 
            settings.VLLM_URL, 
            settings.VLLM_MODEL
        )

if __name__ == "__main__":
    unittest.main()
