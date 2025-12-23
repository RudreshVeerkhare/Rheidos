from pathlib import Path
from rheidos.engine import Engine
from rheidos.scene_config import load_scene_from_config

def main():
    cfg_path = Path("/Users/codebox/dev/kung_fu_panda/apps/point_vortex/scene_configs/vortex.yaml")
    eng = Engine(window_title="Vortex Tutorial", interactive=False)
    cfg = load_scene_from_config(eng, cfg_path)
    eng.start()

if __name__ == "__main__":
    main()