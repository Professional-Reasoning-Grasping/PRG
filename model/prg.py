from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BitsAndBytesConfig, CLIPVisionModel

from utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                         DEFAULT_IMAGE_PATCH_TOKEN)

from .llava.model.language_model.llava_llama import (LlavaLlamaForCausalLM,
                                                     LlavaLlamaModel)
from .segment_anything import build_sam_vit_h

from .segment_anything.modeling.mask_decoder import GraspDecoder


def sigmoid_ce_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    loss = loss.flatten(1, 2).mean(1).sum() / (num_masks + 1e-8)
    return loss


class PRGMetaModel:
    def __init__(
        self,
        config,
        **kwargs,
    ):
        super(PRGMetaModel, self).__init__(config)

        self.config = config
        if not hasattr(self.config, "train_grasp_decoder"):
            self.config.train_grasp_decoder = kwargs["train_grasp_decoder"]
            self.config.out_dim = kwargs["out_dim"]
            self.vision_pretrained = kwargs.get("vision_pretrained", None)
        else:
            self.vision_pretrained = kwargs.get("vision_pretrained", None)
            self.initialize_prg_modules(self.config)

    def initialize_prg_modules(self, config):
        # SAM
        self.visual_model = build_sam_vit_h(self.vision_pretrained)
        for param in self.visual_model.parameters():
            param.requires_grad = False
        if config.train_grasp_decoder:
            self.visual_model.grasp_decoder.train()
            for param in self.visual_model.grasp_decoder.parameters():
                param.requires_grad = True

        grasp_decoder_params = {
            'transformer_dim': self.visual_model.grasp_decoder.transformer_dim, 
            'transformer': self.visual_model.grasp_decoder.transformer,
            'num_multimask_outputs': self.visual_model.grasp_decoder.num_multimask_outputs, 
            'activation': self.visual_model.grasp_decoder.activation,  
            'iou_head_depth': self.visual_model.grasp_decoder.iou_head_depth,  
            'iou_head_hidden_dim': self.visual_model.grasp_decoder.iou_head_hidden_dim,
        }

        self.grasp_decoder = GraspDecoder(**grasp_decoder_params)

        decoder_state_dict = self.visual_model.grasp_decoder.state_dict()

        grasp_decoder_state_dict = self.grasp_decoder.state_dict()

        compatible_state_dict = {k: v for k, v in decoder_state_dict.items() if k in grasp_decoder_state_dict}

        grasp_decoder_state_dict.update(compatible_state_dict)

        self.grasp_decoder.load_state_dict(grasp_decoder_state_dict)


        # Projection layer
        in_dim = config.hidden_size
        out_dim = config.out_dim
        text_fc = [
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, out_dim),
            nn.Dropout(0.0),
        ]
        self.text_hidden_fcs = nn.ModuleList([nn.Sequential(*text_fc)])
        self.text_hidden_fcs.train()
        for param in self.text_hidden_fcs.parameters():
            param.requires_grad = True


class PRGModel(PRGMetaModel, LlavaLlamaModel):
    def __init__(
        self,
        config,
        **kwargs,
    ):
        super(PRGModel, self).__init__(config, **kwargs)

        self.config.use_cache = False
        self.config.vision_tower = self.config.mm_vision_tower
        self.config.mm_vision_select_feature = "patch"
        self.config.image_aspect_ratio = "square"
        self.config.image_grid_pinpoints = None
        self.config.tune_mm_mlp_adapter = False
        self.config.freeze_mm_mlp_adapter = True
        self.config.pretrain_mm_mlp_adapter = None
        self.config.mm_use_im_patch_token = False


class PRGForCausalLM(LlavaLlamaForCausalLM):
    def __init__(
        self,
        config,
        **kwargs,
    ):
        if not hasattr(config, "train_grasp_decoder"):
            config.mm_use_im_start_end = kwargs.pop("use_mm_start_end", True)
            config.mm_vision_tower = kwargs.get(
                "vision_tower", "openai/clip-vit-large-patch14"
            )
            self.ce_loss_weight = kwargs.pop("ce_loss_weight", None)
            self.pos_loss_weight = kwargs.pop("pos_loss_weight", None)
            self.cos_loss_weight = kwargs.pop("cos_loss_weight", None)
            self.sin_loss_weight = kwargs.pop("sin_loss_weight", None)
            self.width_loss_weight = kwargs.pop("width_loss_weight", None)

        else:
            config.mm_vision_tower = config.vision_tower
            
        self.pose_token_idx = kwargs.pop("pose_token_idx")

        super().__init__(config)

        self.model = PRGModel(config, **kwargs)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.post_init()

    def get_visual_embs(self, pixel_values: torch.FloatTensor):
        with torch.no_grad():
            image_embeddings_list = []
            for i in range(pixel_values.shape[0]):
                torch.cuda.empty_cache()
                image_embeddings = self.model.visual_model.image_encoder(
                    pixel_values[i].unsqueeze(0)
                )
                image_embeddings_list.append(image_embeddings)
            torch.cuda.empty_cache()
            image_embeddings = torch.cat(image_embeddings_list, 0)
        return image_embeddings

    def forward(self, **kwargs):
        if "past_key_values" in kwargs:
            return super().forward(**kwargs)
        return self.model_forward(**kwargs)

    def model_forward(
        self,
        images: torch.FloatTensor,
        images_clip: torch.FloatTensor,
        input_ids: torch.LongTensor,
        labels: torch.LongTensor,
        attention_masks: torch.LongTensor,
        offset: torch.LongTensor,
        poses_list: List[torch.FloatTensor],
        label_list: List[torch.Tensor],
        resize_list: List[tuple],
        inference: bool = False,
        **kwargs,
    ):
        image_embeddings = self.get_visual_embs(images)
        batch_size = image_embeddings.shape[0]
        assert batch_size == len(offset) - 1

        pose_token_mask = input_ids[:, 1:] == self.pose_token_idx
        pose_token_mask = torch.cat(
            [
                pose_token_mask,
                torch.zeros((pose_token_mask.shape[0], 1)).bool().cuda(),
            ],
            dim=1,
        )
        pose_token_mask = torch.cat(
            [torch.zeros((pose_token_mask.shape[0], 255)).bool().cuda(), pose_token_mask],
            dim=1,
        )

        if inference:
            n_batch = 1
            length = input_ids.shape[0]
            assert images_clip.shape[0] == 1
            images_clip_extend = images_clip.expand(length, -1, -1, -1).contiguous()

            output_hidden_states = []
            for i in range(n_batch):
                start_i, end_i = i * length, min((i + 1) * length, input_ids.shape[0])
                output_i = super().forward(
                    images=images_clip_extend[: end_i - start_i],
                    attention_mask=attention_masks[start_i:end_i],
                    input_ids=input_ids[start_i:end_i],
                    output_hidden_states=True,
                )
                output_hidden_states.append(output_i.hidden_states)
                torch.cuda.empty_cache()

            output_hidden_states_list = []
            output_hidden_states_level = torch.cat(output_hidden_states, dim=0)
            output_hidden_states_list.append(output_hidden_states_level)
            output_hidden_states = output_hidden_states_list
            output = None

        else:
            images_clip_list = []
            for i in range(len(offset) - 1):
                start_i, end_i = offset[i], offset[i + 1]
                images_clip_i = (
                    images_clip[i]
                    .unsqueeze(0)
                    .expand(end_i - start_i, -1, -1, -1)
                    .contiguous()
                )
                images_clip_list.append(images_clip_i)
            images_clip = torch.cat(images_clip_list, dim=0)

            output = super().forward(
                images=images_clip,
                attention_mask=attention_masks,
                input_ids=input_ids,
                labels=labels,
                output_hidden_states=True,
            )
            output_hidden_states = output.hidden_states

        hidden_states = []

        assert len(self.model.text_hidden_fcs) == 1
        hidden_states.append(self.model.text_hidden_fcs[0](output_hidden_states[-1]))

        last_hidden_state = torch.stack(hidden_states, dim=-1).sum(dim=-1)
        pred_embeddings = last_hidden_state[pose_token_mask]
        pose_token_counts = pose_token_mask.int().sum(-1)  # [bs, ]

        pose_token_offset = pose_token_counts.cumsum(-1)
        pose_token_offset = torch.cat(
            [torch.zeros(1).long().cuda(), pose_token_offset], dim=0
        )

        pose_token_offset = pose_token_offset[offset]

        pred_embeddings_ = []
        for i in range(len(pose_token_offset) - 1):
            start_i, end_i = pose_token_offset[i], pose_token_offset[i + 1]
            pred_embeddings_.append(pred_embeddings[start_i:end_i])
        pred_embeddings = pred_embeddings_

        multimask_output = False
        pos_output_list = []
        cos_output_list = []
        sin_output_list = []
        width_output_list = []
        for i in range(len(pred_embeddings)):
            (
                sparse_embeddings,
                dense_embeddings,
            ) = self.model.visual_model.prompt_encoder(
                points=None,
                boxes=None,
                masks=None,
                text_embeds=pred_embeddings[i].unsqueeze(1),
            )
            sparse_embeddings = sparse_embeddings.to(pred_embeddings[i].dtype)

            grasp_output, iou_predictions = self.model.grasp_decoder(
                image_embeddings=image_embeddings[i].unsqueeze(0),
                image_pe=self.model.visual_model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output,
            )
            

            pos_output = self.model.visual_model.postprocess_grasp(
                grasp_output[0],
                input_size=resize_list[i],
                original_size=label_list[i].shape,
            )

            cos_output = self.model.visual_model.postprocess_grasp(
                grasp_output[1],
                input_size=resize_list[i],
                original_size=label_list[i].shape,
            )

            sin_output = self.model.visual_model.postprocess_grasp(
                grasp_output[2],
                input_size=resize_list[i],
                original_size=label_list[i].shape,
            )

            width_output = self.model.visual_model.postprocess_grasp(
                grasp_output[3],
                input_size=resize_list[i],
                original_size=label_list[i].shape,
            )

            pos_output_list.append(pos_output)
            cos_output_list.append(cos_output)
            sin_output_list.append(sin_output)
            width_output_list.append(width_output)

        model_output = output
        gt_grasp = poses_list

        if inference:
            return {"pos_outputs": pos_output_list, "cos_outputs": cos_output_list, "sin_outputs": sin_output_list, 
                    "width_outputs": width_output_list,"gt_grasp":gt_grasp}

        gt_poss = gt_grasp[0]
        gt_coss = gt_grasp[1]
        gt_sins = gt_grasp[2]
        gt_widths = gt_grasp[3]

        output = model_output.logits

        ce_loss = model_output.loss
        ce_loss = ce_loss * self.ce_loss_weight

        p_loss = 0
        cos_loss = 0
        sin_loss = 0
        width_loss = 0

        num_masks = 0

        for batch_idx in range(len(pos_output_list)):
            gt_pos = gt_poss[batch_idx]
            gt_cos = gt_coss[batch_idx]
            gt_sin = gt_sins[batch_idx]
            gt_width = gt_widths[batch_idx]

            pred_pos = pos_output_list[batch_idx]
            pred_cos = cos_output_list[batch_idx]
            pred_sin = sin_output_list[batch_idx]
            pred_width = width_output_list[batch_idx]

            p_loss += (
                F.smooth_l1_loss(pred_pos, gt_pos)
            )
            cos_loss += (
                F.smooth_l1_loss(pred_cos,gt_cos)
            )
            sin_loss += (
               F.smooth_l1_loss(pred_sin, gt_sin)
            )
            width_loss += (
                F.smooth_l1_loss(pred_width, gt_width)
            )
            num_masks += gt_pos.shape[0]

        p_loss = self.pos_loss_weight * p_loss
        cos_loss = self.cos_loss_weight * cos_loss
        sin_loss = self.sin_loss_weight * sin_loss
        width_loss = self.width_loss_weight * width_loss

        grasp_loss = p_loss + cos_loss + sin_loss + width_loss

        loss = ce_loss + grasp_loss

        return {
            "loss": loss,
            "ce_loss": ce_loss,
            "grasp_loss": grasp_loss,
            "p_loss": p_loss,
            "cos_loss": cos_loss,
            "sin_loss": sin_loss,
            "width_loss": width_loss,
        }

    def evaluate(
        self,
        images_clip,
        images,
        input_ids,
        resize_list,
        original_size_list,
        max_new_tokens=32,
        tokenizer=None,
    ):
        with torch.no_grad():
            outputs = self.generate(
                images=images_clip,
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                num_beams=1,
                output_hidden_states=True,
                return_dict_in_generate=True,
            )
            output_hidden_states = outputs.hidden_states[-1]
            output_ids = outputs.sequences

            pose_token_mask = output_ids[:, 1:] == self.pose_token_idx
            
            pose_token_mask = torch.cat(
                [
                    torch.zeros((pose_token_mask.shape[0], 255)).bool().cuda(),
                    pose_token_mask,
                ],
                dim=1,
            )

            hidden_states = []

            assert len(self.model.text_hidden_fcs) == 1
            hidden_states.append(self.model.text_hidden_fcs[0](output_hidden_states))

            last_hidden_state = torch.stack(hidden_states, dim=-1).sum(dim=-1)
            pred_embeddings = last_hidden_state[pose_token_mask]

            pose_token_counts = pose_token_mask.int().sum(-1)  # [bs, ]
            pose_token_offset = pose_token_counts.cumsum(-1)
            pose_token_offset = torch.cat(
                [torch.zeros(1).long().cuda(), pose_token_offset], dim=0
            )

            pred_embeddings_ = []
            for i in range(len(pose_token_offset) - 1):
                start_i, end_i = pose_token_offset[i], pose_token_offset[i + 1]
                pred_embeddings_.append(pred_embeddings[start_i:end_i])
            pred_embeddings = pred_embeddings_

            image_embeddings = self.get_visual_embs(images)

            multimask_output = False
            pos_output_list = []
            cos_output_list = []
            sin_output_list = []
            width_output_list = []
            for i in range(len(pred_embeddings)):
                (
                    sparse_embeddings,
                    dense_embeddings,
                ) = self.model.visual_model.prompt_encoder(
                    points=None,
                    boxes=None,
                    masks=None,
                    text_embeds=pred_embeddings[i].unsqueeze(1),
                )

                sparse_embeddings = sparse_embeddings.to(pred_embeddings[i].dtype)
                grasp_output, iou_predictions = self.model.grasp_decoder(
                    image_embeddings=image_embeddings[i].unsqueeze(0),
                    image_pe=self.model.visual_model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=multimask_output,
                )
                pos_output = self.model.visual_model.postprocess_grasp(
                grasp_output[0],
                input_size=resize_list[i],
                original_size=original_size_list[i].shape,
                )

                cos_output = self.model.visual_model.postprocess_grasp(
                    grasp_output[1],
                    input_size=resize_list[i],
                    original_size=original_size_list[i].shape,
                )

                sin_output = self.model.visual_model.postprocess_grasp(
                    grasp_output[2],
                    input_size=resize_list[i],
                    original_size=original_size_list[i].shape,
                )

                width_output = self.model.visual_model.postprocess_grasp(
                    grasp_output[3],
                    input_size=resize_list[i],
                    original_size=original_size_list[i].shape,
                )
                pos_output_list.append(pos_output)
                cos_output_list.append(cos_output)
                sin_output_list.append(sin_output)
                width_output_list.append(width_output)

        pred_grasp = [pos_output_list, cos_output_list, sin_output_list, width_output_list]
        return output_ids, pred_grasp