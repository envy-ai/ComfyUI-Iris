from comfy_api.latest import ComfyExtension

from .nodes import IrisModelLoader, IrisCLIPLoader, IrisKSampler, IrisSamplerCustomAdvanced


class IrisExtension(ComfyExtension):
    async def get_node_list(self):
        return [IrisModelLoader, IrisCLIPLoader, IrisKSampler, IrisSamplerCustomAdvanced]


async def comfy_entrypoint():
    return IrisExtension()
