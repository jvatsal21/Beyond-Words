import torch
import torch.nn as nn

class QwenVLRegressionModel(nn.Module):
    def __init__(self, vlm):
        super().__init__()
        self.vlm = vlm
        hidden_size = vlm.config.hidden_size
        
        # Use the text embedding layer.
        self.text_embed = vlm.get_input_embeddings()
        
        # For memory: we reduce image resolution from 224x224 to e.g. 112x112.
        self.image_resolution = 112  
        self.num_channels = 6  
        flat_size = self.num_channels * self.image_resolution * self.image_resolution
        
        self.image_proj = nn.Sequential(
            nn.Linear(flat_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU()
        )
        
        self.regressor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        ).float()

    def forward(self, pixel_values, input_ids, attention_mask, labels=None):
        # Assume pixel_values now has shape (batch, channels, self.image_resolution, self.image_resolution)
        batch_size = input_ids.shape[0]
        
        # Process image pixels:
        flat_images = pixel_values.view(batch_size, -1)

        flat_images = flat_images.half()
        image_embeds = self.image_proj(flat_images)
        image_embeds = image_embeds.unsqueeze(1)
        
        # Get text embeddings.
        text_embeds = self.text_embed(input_ids)
        
        # Combine by prepending image embedding.
        inputs_embeds = torch.cat([image_embeds, text_embeds], dim=1)
        
        # Create combined attention mask.
        image_mask = torch.ones(batch_size, 1, dtype=attention_mask.dtype, device=attention_mask.device)
        combined_mask = torch.cat([image_mask, attention_mask], dim=1)
        
        # Forward pass through the model.
        outputs = self.vlm(
            inputs_embeds=inputs_embeds,
            attention_mask=combined_mask,
            return_dict=True,
        )
        
        # Extract the output for the first token (assumed to represent combined info).
        cls_emb = outputs.last_hidden_state[:, 0, :].float()
        preds = self.regressor(cls_emb).squeeze(-1)
        
        loss = None
        if labels is not None:
            loss = nn.HuberLoss()(preds, labels.float())
        
        return {"loss": loss, "preds": preds}
