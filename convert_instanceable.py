import omni.client
import omni.usd
from pxr import Sdf, UsdGeom, UsdShade


def create_parent_xforms(asset_usd_path, source_prim_path):
    """Adds a new UsdGeom.Xform prim for each Mesh/Geometry prim under source_prim_path.
    Moves material assignment to new parent prim if any exists on the Mesh/Geometry prim.

    Args:
        asset_usd_path (str): USD file path for asset
        source_prim_path (str): USD path of root prim
    """
    omni.usd.get_context().open_stage(asset_usd_path)
    stage = omni.usd.get_context().get_stage()

    prims = [stage.GetPrimAtPath(source_prim_path)]
    edits = Sdf.BatchNamespaceEdit()
    while len(prims) > 0:
        prim = prims.pop(0)
        if prim.GetTypeName() in ["Mesh", "Capsule", "Sphere", "Box", "Cylinder", "Cube"]:
            new_xform = UsdGeom.Xform.Define(stage, str(prim.GetPath()) + "_xform")
            print(prim, new_xform)
            edits.Add(Sdf.NamespaceEdit.Reparent(prim.GetPath(), new_xform.GetPath(), 0))
            # get material binding
            material_binding = UsdShade.MaterialBindingAPI(prim)
            material = material_binding.ComputeBoundMaterial()[0]
            # unbind material if exists
            if material:
                material_binding.UnbindAllBindings()
                # bind material to new parent
                material_binding = UsdShade.MaterialBindingAPI(new_xform)
                material_binding.Bind(material)
            continue

        children_prims = prim.GetChildren()
        prims = prims + children_prims

    stage.GetRootLayer().Apply(edits)
    omni.usd.get_context().save_stage()


def convert_asset_instanceable(asset_usd_path, source_prim_path, create_xforms=False):
    """Makes all mesh/geometry prims instanceable.
    Makes a copy of the asset USD file, which will be used for referencing.
    Updates asset file to convert all parent prims of mesh/geometry prims to reference cloned USD file.

    Args:
        asset_usd_path (str): USD file path for asset
        source_prim_path (str): USD path of root prim
        save_as_path (str): USD file path for modified USD stage. Defaults to None, will save in same file.
    """
    if create_xforms:
        create_parent_xforms(asset_usd_path, source_prim_path)

    instance_usd_path = asset_usd_path.replace(".usd", "_instanceable.usd")
    omni.client.copy(asset_usd_path, instance_usd_path)
    omni.usd.get_context().open_stage(instance_usd_path)
    stage = omni.usd.get_context().get_stage()

    prims = [stage.GetPrimAtPath(source_prim_path)]
    while len(prims) > 0:
        prim = prims.pop(0)
        if prim:
            if prim.GetTypeName() in ["Mesh", "Capsule", "Sphere", "Box", "Cylinder", "Cube"]:
                parent_prim = prim.GetParent()
                if parent_prim and not parent_prim.IsInstance():
                    parent_prim.GetReferences().AddReference(
                        assetPath=asset_usd_path, primPath=str(parent_prim.GetPath())
                    )
                    parent_prim.SetInstanceable(True)
                    continue

            children_prims = prim.GetChildren()
            prims = prims + children_prims

    omni.usd.get_context().save_stage()


base_path = "/home/svl/Documents/OmniGibson/omnigibson/data/assets/models"
asset_usd_path = "tiago/tiago_dual_omnidirectional_stanford/tiago_dual_omnidirectional_stanford_33.usd"


convert_asset_instanceable(
    asset_usd_path=f"{base_path}/{asset_usd_path}", source_prim_path="/tiago_dual", create_xforms=True
)
