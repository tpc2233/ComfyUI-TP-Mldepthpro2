from comfy_api.latest import ComfyExtension

from .nodes import TPDepthPro2Loader, TPDepthPro2Video, TPDepthPro2Preview, TPDepthPro2RawImage, TPDepthPro2SaveDepth


class TPDepthPro2Extension(ComfyExtension):
    async def get_node_list(self):
        return [TPDepthPro2Loader, TPDepthPro2Video, TPDepthPro2Preview, TPDepthPro2RawImage, TPDepthPro2SaveDepth]


async def comfy_entrypoint():
    return TPDepthPro2Extension()
