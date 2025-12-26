from pathlib import Path

from rheidos.engine import Engine
from rheidos.scene_config import load_scene_from_config

from rheidos.controllers.point_selector import SceneVertexPointSelector
from rheidos.views import PointSelectionView
from panda3d.core import BitMask32


from rheidos.compute.world import World


def main() -> None:
    cfg_path = Path(__file__).resolve().parent / "scene_configs" / "poisson.yaml"
    eng = Engine(window_title="Poisson Mesh Demo", interactive=False)
    load_scene_from_config(eng, cfg_path)

    pick_mask = BitMask32.bit(4)

    pos_markers = PointSelectionView(
        name="pos_markers",
        selected_color=(1.0, 0.25, 0.25, 1.0),
        hover_color=(1.0, 0.9, 0.5, 1.0),
    )
    neg_markers = PointSelectionView(
        name="neg_markers",
        selected_color=(0.2, 0.4, 1.0, 1.0),
        hover_color=(0.6, 0.8, 1.0, 1.0),
    )

    eng.add_view(pos_markers)
    eng.add_view(neg_markers)

    pos_selector = SceneVertexPointSelector(
        engine=eng,
        pick_mask=pick_mask,
        markers_view=pos_markers,
        store_key="poisson/pos_points",
        select_button="mouse1",
        clear_shortcut="c",
    )
    neg_selector = SceneVertexPointSelector(
        engine=eng,
        pick_mask=pick_mask,
        markers_view=neg_markers,
        store_key="poisson/neg_points",
        select_button="mouse3",
        clear_shortcut="v",
    )

    eng.add_controller(pos_selector)
    eng.add_controller(neg_selector)



    eng.start()


if __name__ == "__main__":
    main()
