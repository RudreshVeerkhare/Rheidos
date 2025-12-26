from pathlib import Path
from panda3d.core import BitMask32
from rheidos.engine import Engine
from rheidos.scene_config import load_scene_from_config
from rheidos.views import PointSelectionView
from rheidos.controllers.point_selector import SceneSurfacePointSelector





def main():
    cfg_path = Path("/Users/codebox/dev/kung_fu_panda/apps/point_vortex/scene_configs/vortex.yaml")
    eng = Engine(window_title="Vortex Tutorial", interactive=False)
    cfg = load_scene_from_config(eng, cfg_path)

    markers = PointSelectionView(name="seed_markers")
    eng.add_view(markers)
    eng.add_controller(
        SceneSurfacePointSelector(
            engine=eng,
            pick_mask=BitMask32.bit(4),
            markers_view=markers,
            store_key="vortex/seed_points"
        )
    )

    eng.start()

if __name__ == "__main__":
    main()