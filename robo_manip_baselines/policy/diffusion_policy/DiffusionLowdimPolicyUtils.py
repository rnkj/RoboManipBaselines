def construct_lowdim_policy(noise_scheduler, backbone, policy_args, model_args):
    if backbone == "cnn":
        from diffusion_policy.model.diffusion.conditional_unet1d import (
            ConditionalUnet1D,
        )
        from diffusion_policy.policy.diffusion_unet_lowdim_policy import (
            DiffusionUnetLowdimPolicy,
        )

        model = ConditionalUnet1D(**model_args)
        policy = DiffusionUnetLowdimPolicy(
            model=model,
            noise_scheduler=noise_scheduler,
            **policy_args,
        )
    elif backbone == "transformer":
        from diffusion_policy.model.diffusion.transformer_for_diffusion import (
            TransformerForDiffusion,
        )
        from diffusion_policy.policy.diffusion_transformer_lowdim_policy import (
            DiffusionTransformerLowdimPolicy,
        )

        model = TransformerForDiffusion(**model_args)
        policy = DiffusionTransformerLowdimPolicy(
            model=model,
            noise_scheduler=noise_scheduler,
            **policy_args,
        )
    else:
        raise ValueError(f"Invalid backbone: {backbone}")

    setup_lowdim_identity_normalizer(policy)
    return policy


def setup_lowdim_identity_normalizer(policy):
    from diffusion_policy.model.common.normalizer import (
        LinearNormalizer,
        SingleFieldLinearNormalizer,
    )

    normalizer = LinearNormalizer()
    normalizer["obs"] = SingleFieldLinearNormalizer.create_identity()
    normalizer["action"] = SingleFieldLinearNormalizer.create_identity()
    policy.set_normalizer(normalizer)
