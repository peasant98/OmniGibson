import omnigibson as og
from omnigibson.object_states.object_state_base import AbsoluteObjectState
from omnigibson.utils.sim_utils import prims_to_rigid_prim_set


class ContactBodies(AbsoluteObjectState):

    def _get_value(self, ignore_objs=None):
        # Compute bodies in contact, minus the self-owned bodies
        bodies = set()
        for contact in self.obj.contact_list():
            bodies.update({contact.body0, contact.body1})
        bodies -= set(self.obj.link_prim_paths)
        bodies = {og.sim.scene.object_registry("prim_path", "/".join(body.split("/")[:-1])).links[body.split("/")[-1]] for body in bodies}
        # Ignore_objs should either be None or tuple (CANNOT be list because we need to hash these inputs)
        assert ignore_objs is None or isinstance(ignore_objs, tuple), \
            "ignore_objs must either be None or a tuple of objects to ignore!"
        return bodies if ignore_objs is None else bodies - prims_to_rigid_prim_set(ignore_objs)

    def _set_value(self, new_value):
        raise NotImplementedError("ContactBodies state currently does not support setting.")
