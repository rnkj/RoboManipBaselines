from collections.abc import Iterable

import torch
import torch.nn as nn
from eipl.layer import InverseSpatialSoftmax, SpatialSoftmax


class SwiGLU(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.gate = nn.SiLU(inplace=True)

        
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.GELU()

    def forward(self, x):
        return self.activation(self.linear1(x)) * self.linear2(x)

class KaSarnnPolicy(nn.Module):
    """
    KASARNN (Spatial attention recurrent neural network) policy.

    This implementation extends the original SARNN so that it can handle multiple images and additional data such as wrench. See the following repository for the original SARNN implementation: https://github.com/ogata-lab/eipl
    """

    def __init__(
        self,
        state_dim,
        num_images,
        image_size_list,
        num_attentions,
        num_classes_list,
        lstm_hidden_dim,
        adapter_hidden_dim,
    ):
        # Setup variables
        self.num_attentions = num_attentions
        self.num_classes_list = num_classes_list.copy()

        kernel_size = 3
        softmax_temperature = 1e-4
        inv_softmax_heatmap_size = 0.1

        if isinstance(image_size_list[0], Iterable):
            assert num_images == len(image_size_list)
        else:
            image_size_list = [image_size_list] * num_images

        super().__init__()

        # Instantiate layers
        activation = nn.LeakyReLU(negative_slope=0.3)
        encoded_image_size_list = [
            (
                image_size[0] - 3 * (kernel_size - 1),
                image_size[1] - 3 * (kernel_size - 1),
            )
            for image_size in image_size_list
        ]

        self.attention_encoder_list = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(3, 16, kernel_size, 1, 0),
                activation,
                nn.Conv2d(16, 32, kernel_size, 1, 0),
                activation,
                nn.Conv2d(32, self.num_attentions, kernel_size, 1, 0),
                activation,
                SpatialSoftmax(
                    width=encoded_image_size[0],
                    height=encoded_image_size[1],
                    temperature=softmax_temperature,
                    normalized=True,
                ),
            )
            for encoded_image_size in encoded_image_size_list
        )

        self.image_encoder_list = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(3, 16, kernel_size, 1, 0),
                activation,
                nn.Conv2d(16, 32, kernel_size, 1, 0),
                activation,
                nn.Conv2d(32, self.num_attentions, kernel_size, 1, 0),
                activation,
            )
            for _ in range(num_images)
        )

        lstm_input_dim = state_dim + num_images * self.num_attentions * 2
        adapter_upscale_dim = len(num_classes_list) * adapter_hidden_dim
        self.knowledge_adapters = nn.ModuleDict(
            {
                "rnn": nn.LSTMCell(lstm_input_dim, lstm_input_dim),
                "up_proj": nn.Linear(
                    lstm_input_dim, adapter_upscale_dim, bias=False
                ),
                "gate": nn.Sequential(
                    nn.Linear(lstm_input_dim, adapter_upscale_dim, bias=False),
                    nn.SiLU(inplace=True),
                ),
                "down_proj": nn.Linear(adapter_upscale_dim, lstm_input_dim, bias=False),
                "classifier": nn.ModuleList(
                    [
                        nn.Linear(adapter_hidden_dim, num_classes)
                        for num_classes in num_classes_list
                    ]
                ),
            }
        )

        self.lstm = nn.LSTMCell(lstm_input_dim, lstm_hidden_dim)

        self.state_decoder = nn.Sequential(
            nn.Linear(lstm_hidden_dim, state_dim),
            # activation,
        )

        self.attention_decoder_list = nn.ModuleList(
            nn.Sequential(
                nn.Linear(lstm_hidden_dim, self.num_attentions * 2),
                activation,
            )
            for _ in range(num_images)
        )

        self.inv_softmax_list = nn.ModuleList(
            InverseSpatialSoftmax(
                width=encoded_image_size[0],
                height=encoded_image_size[1],
                heatmap_size=inv_softmax_heatmap_size,
                normalized=True,
            )
            for encoded_image_size in encoded_image_size_list
        )

        self.image_decoder_list = nn.ModuleList(
            nn.Sequential(
                nn.ConvTranspose2d(self.num_attentions, 32, kernel_size, 1, 0),
                activation,
                nn.ConvTranspose2d(32, 16, kernel_size, 1, 0),
                activation,
                nn.ConvTranspose2d(16, 3, kernel_size, 1, 0),
                # activation,
            )
            for _ in range(num_images)
        )

        # Initialize weights
        self.apply(self._initialize_weights)

    def _initialize_weights(self, m):
        if isinstance(m, nn.LSTMCell):
            nn.init.xavier_uniform_(m.weight_ih)
            nn.init.orthogonal_(m.weight_hh)
            nn.init.zeros_(m.bias_ih)
            nn.init.zeros_(m.bias_hh)
        elif (
            isinstance(m, nn.Conv2d)
            or isinstance(m, nn.ConvTranspose2d)
            or isinstance(m, nn.Linear)
        ):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, state, image_list, lstm_state=None, adapter_state=None):
        assert len(image_list) == len(self.image_encoder_list)

        # Encode images
        encoded_image_list = []
        for image, image_encoder in zip(image_list, self.image_encoder_list):
            encoded_image_list.append(image_encoder(image))

        # Calculate encoded attentions
        attention_list = []
        for image, attention_encoder in zip(image_list, self.attention_encoder_list):
            attention = attention_encoder(image)[0].reshape(-1, self.num_attentions * 2)
            attention_list.append(attention)

        # Forward knowledge adapter
        lstm_input = torch.cat([state, *attention_list], dim=-1)
        new_adapter_state = self.knowledge_adapters["rnn"](lstm_input, adapter_state)
        up = self.knowledge_adapters["up_proj"](new_adapter_state[0])
        gate = self.knowledge_adapters["gate"](lstm_input)
        knowledge = self.knowledge_adapters["down_proj"](up * gate)

        # Classify knowledge from `up`
        up_chunks = torch.chunk(up, len(self.num_classes_list), dim=-1)
        logits = [self.knowledge_adapters["classifier"][i](up_chunk) for i, up_chunk in enumerate(up_chunks)]
        
        # Forward LSTM
        lstm_input = lstm_input + knowledge
        lstm_output = self.lstm(lstm_input, lstm_state)

        # Decode state
        predicted_state = self.state_decoder(lstm_output[0])

        # Decode attentions
        predicted_attention_list = []
        for attention_decoder in self.attention_decoder_list:
            predicted_attention = attention_decoder(lstm_output[0]).reshape(
                -1, self.num_attentions, 2
            )
            predicted_attention_list.append(predicted_attention)

        # Decode images
        predicted_image_list = []
        for encoded_image, predicted_attention, inv_softmax, image_decoder in zip(
            encoded_image_list,
            predicted_attention_list,
            self.inv_softmax_list,
            self.image_decoder_list,
        ):
            heatmap = inv_softmax(predicted_attention)
            predicted_image = image_decoder(torch.mul(heatmap, encoded_image))
            predicted_image_list.append(predicted_image)

        # Post-process
        attention_list = [
            attention.reshape(-1, self.num_attentions, 2)
            for attention in attention_list
        ]

        return (
            predicted_state,  # (batch_size, state_dim)
            predicted_image_list,  # (num_images, batch_size, 3, width, height)
            attention_list,  # (num_images, batch_size, num_attentions, 2)
            predicted_attention_list,  # (num_images, batch_size, num_attentions, 2)
            logits,  # (num_agents, batch_size, num_classes[i])
            lstm_output,
            new_adapter_state,
        )
