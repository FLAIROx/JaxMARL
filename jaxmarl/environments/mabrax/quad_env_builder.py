# quad_env_builder.py

import math
from xml.etree import ElementTree as ET
from xml.dom import minidom
from typing import List, Optional
import colorsys


class QuadEnvGenerator:
    """
    Generates a MuJoCo XML for N quadrotors tethered to a central payload,
    with a built-in 'track' camera for rendering.
    """

    def __init__(
        self,
        n_quads: int,
        cable_length: float = 0.5,
        tendon_width: float = 0.001,
        payload_height: float = 1.5,
        payload_mass: float = 0.01,
        frame_radius: float = 0.15,
        mesh_dir: str = "jaxmarl/environments/mabrax/mujoco/assets",
        mesh_names: Optional[List[str]] = None,
        offscreen_width: int = 1920,
        offscreen_height: int = 1080,
        vis_azimuth: float = -20,
        vis_elevation: float = -20,
        camera_pos: str = "-0.7 0 0.5",
        camera_quat: str = "0.601501 0.371748 -0.371748 -0.601501",
        camera_mode: str = "trackcom",
    ):
        self.n = n_quads
        self.cable_length = cable_length
        self.tendon_width = tendon_width
        self.payload_height = payload_height
        self.payload_mass = payload_mass
        self.frame_radius = frame_radius
        self.mesh_dir = mesh_dir
        self.mesh_names = mesh_names or [f"cf2_{i}" for i in range(7)]
        self.offscreen_width = offscreen_width
        self.offscreen_height = offscreen_height
        self.vis_azimuth = vis_azimuth
        self.vis_elevation = vis_elevation
        # camera settings
        self.camera_pos = camera_pos
        self.camera_quat = camera_quat
        self.camera_mode = camera_mode

        # Print out all the Environment parameters
        print(f"Generating XML for {self.n} quadrotors with payload:")
        print(f"  Cable Length: {self.cable_length} m")
        print(f"  Tendon Width: {self.tendon_width} m")
        print(f"  Payload Height: {self.payload_height} m")
        print(f"  Payload Mass: {self.payload_mass} kg")
        print(f"  Frame Radius: {self.frame_radius} m")
        print(f"  Mesh Directory: {self.mesh_dir}")
               

    def _prettify(self, elem: ET.Element) -> str:
        rough = ET.tostring(elem, "utf-8")
        return minidom.parseString(rough).toprettyxml(indent="    ")

    def generate_xml(self) -> str:
        mj = ET.Element("mujoco", {"model": f"n{self.n}_quads_payload"})

        # compiler & option
        ET.SubElement(mj, "compiler", {
            "angle": "radian", "meshdir": self.mesh_dir, "discardvisual": "false"
        })
        opt = ET.SubElement(mj, "option", {
            "timestep": "0.004",
            "gravity": "0 0 -9.81",
            "solver": "Newton",
            "jacobian": "dense",
            "iterations": "1",
            "ls_iterations": "2",
            "integrator": "Euler",
        })
        ET.SubElement(opt, "flag", {"eulerdamp": "disable"})

        # visual (offscreen framebuffer)
        vis = ET.SubElement(mj, "visual")
        ET.SubElement(vis, "global", {
            "azimuth": str(self.vis_azimuth),
            "elevation": str(self.vis_elevation),
            "ellipsoidinertia": "true",
            "offwidth": str(self.offscreen_width),
            "offheight": str(self.offscreen_height),
        })
        ET.SubElement(vis, "headlight", {
            "ambient": "0.3 0.3 0.3", "diffuse": "0.6 0.6 0.6", "specular": "0 0 0",
        })
        ET.SubElement(vis, "rgba", {"haze": "0.15 0.25 0.35 1"})
        ET.SubElement(vis, "scale", {"jointlength": "0", "jointwidth": "0"})

        # default classes for cf2
        defaults = ET.SubElement(mj, "default")
        d_cf2 = ET.SubElement(defaults, "default", {"class": "cf2"})
        ET.SubElement(d_cf2, "site", {"group": "5"})
        d_vis = ET.SubElement(d_cf2, "default", {"class": "visual"})
        ET.SubElement(d_vis, "geom", {
            "type": "mesh", "contype": "0", "conaffinity": "0", "group": "2"
        })
        d_col = ET.SubElement(d_cf2, "default", {"class": "collision"})
        ET.SubElement(d_col, "geom", {"type": "mesh", "group": "3"})

        # assets
        asset = ET.SubElement(mj, "asset")
        # textures
        ET.SubElement(asset, "texture", {
            "type": "skybox", "builtin": "gradient",
            "rgb1": "0.3 0.5 0.7", "rgb2": "0 0 0", "width": "512", "height": "3072"
        })
        ET.SubElement(asset, "texture", {
            "type": "2d", "name": "groundplane", "builtin": "checker",
            "mark": "edge", "rgb1": "0.2 0.3 0.4", "rgb2": "0.1 0.2 0.3",
            "markrgb": "0.8 0.8 0.8", "width": "300", "height": "300"
        })
        # materials
        mats = {
            "polished_plastic": "0.631 0.659 0.678 1",
            "polished_gold":    "0.969 0.878 0.6 1",
            "medium_gloss_plastic": "0.109 0.184 0 1",
            "propeller_plastic":    "0.792 0.82 0.933 1",
            "white":                "",
            "body_frame_plastic":   "0.102 0.102 0.102 1",
            "burnished_chrome":     "0.898 0.898 0.898 1",
        }
        for name, rgba in mats.items():
            attrs = {"name": name}
            if rgba:
                attrs["rgba"] = rgba
            ET.SubElement(asset, "material", attrs)
        ET.SubElement(asset, "material", {
            "name": "groundplane", "texture": "groundplane",
            "texuniform": "true", "texrepeat": "10 10", "reflectance": "0"
        })
        # meshes
        for m in self.mesh_names:
            ET.SubElement(asset, "mesh", {"name": m, "file": f"{m}.obj"})

        # worldbody
        wb = ET.SubElement(mj, "worldbody")
        ET.SubElement(wb, "geom", {
            "name": "floor", "type": "plane", "size": "0 0 0.05", "material": "groundplane"
        })
        ET.SubElement(wb, "light", {
            "pos": "0 0 1.5", "dir": "0 0 -1", "directional": "true"
        })

        # payload body + free joint + geom + site
        p = ET.SubElement(wb, "body", {
            "name": "payload", "pos": f"0 0 {self.payload_height}"
        })
        ET.SubElement(p, "joint", {
            "name": "payload_joint", "type": "free",
            "actuatorfrclimited": "false", "damping": "0.00001"
        })
        ET.SubElement(p, "geom", {
            "type": "sphere", "size": "0.01",
            "mass": str(self.payload_mass),
            "rgba": "0.8 0.8 0.8 1"
        })
        ET.SubElement(p, "site", {"name": "payload_s", "pos": "0 0 0.01"})

        # <-- HERE: insert the default 'track' camera -->
        ET.SubElement(p, "camera", {
            "name": "track",
            "pos": self.camera_pos,
            "quat": self.camera_quat,
            "mode": self.camera_mode,
        })

        # N quads
        for i in range(self.n):
            theta = 2 * math.pi * i / self.n
            x = self.frame_radius * math.cos(theta)
            y = self.frame_radius * math.sin(theta)
            base = ET.SubElement(wb, "body", {
                "name": f"q{i}_container",
                "pos": f"{x:.3f} {y:.3f} {self.payload_height + 0.1}"
            })
            ET.SubElement(base, "joint", {
                "name": f"q{i}_joint", "type": "free", "actuatorfrclimited": "false"
            })
            ET.SubElement(base, "site", {
                "name": f"q{i}_attachment", "pos": "0 0 0", "group": "5"
            })
            cf2 = ET.SubElement(base, "body", {
                "name": f"q{i}_cf2", "childclass": "cf2", "pos": "0 0 -0.0015"
            })
            ET.SubElement(cf2, "inertial", {
                "pos": "0 0 0", "mass": "0.034",
                "diaginertia": "1.65717e-05 1.66556e-05 2.92617e-05"
            })
            # visual & collision
            visual_mats = [
                "propeller_plastic", "medium_gloss_plastic", "polished_gold",
                "polished_plastic", "burnished_chrome", "body_frame_plastic",
                "white"
            ]
            for mat, mesh in zip(visual_mats, self.mesh_names):
                ET.SubElement(cf2, "geom", {"class": "visual", "material": mat, "mesh": mesh})
            ET.SubElement(cf2, "geom", {
                "class": "collision", "type": "box", "pos": "0 0 0",
                "size": "0.05 0.05 0.015", "group": "3", "rgba": "1 0 0 0.5"
            })
            ET.SubElement(cf2, "site", {"name": f"q{i}_imu", "pos": "0 0 0"})
            for lbl, sx, sy in [("1",1,-1),("2",-1,-1),("3",-1,1),("4",1,1)]:
                ET.SubElement(cf2, "site", {
                    "name": f"q{i}_thrust{lbl}",
                    "pos": f"{0.032527*sx:.6f} {0.032527*sy:.6f} 0"
                })

        # tendons
        td = ET.SubElement(mj, "tendon")
        # generate hues for distinct tendon colors
        hues = list(i / self.n for i in range(self.n))

        for i in range(self.n):
            h = hues[i]
            r, g, b = colorsys.hsv_to_rgb(h, 0.7, 1.0)
            rgba = f"{r:.3f} {g:.3f} {b:.3f} 1"
            sp = ET.SubElement(td, "spatial", {
                "name": f"q{i}_tendon", "limited": "true",
                "range": f"0 {self.cable_length}",
                "width": str(self.tendon_width), "rgba": rgba
            })
            ET.SubElement(sp, "site", {"site": f"q{i}_attachment"})
            ET.SubElement(sp, "site", {"site": "payload_s"})

        # actuators
        act = ET.SubElement(mj, "actuator")
        for i in range(self.n):
            for lbl in ("1","2","3","4"):
                gear = 6e-06 if int(lbl) % 2 else -6e-06
                ET.SubElement(act, "general", {
                    "name": f"q{i}_thrust{lbl}",
                    "class": "cf2",
                    "site": f"q{i}_thrust{lbl}",
                    "ctrlrange": "0 0.14",
                    "gear": f"0 0 1 0 0 {gear:.0e}"
                })

        # visualize goal as a semi-transparent green sphere
        ET.SubElement(wb, "site", {
            "name": "goal_marker",
            "pos": f"0 0 {self.payload_height}",
            "size": "0.01",
            "type": "sphere",
            "rgba": "0 1 0 0.7"
        })

        return self._prettify(mj)


if __name__ == "__main__":
    # Example usage with default 'track' camera present
    gen = QuadEnvGenerator(n_quads=4)
    xml = gen.generate_xml()
    with open("quad_env_with_camera.xml", "w") as f:
        f.write(xml)
    print("Written quad_env_with_camera.xml")