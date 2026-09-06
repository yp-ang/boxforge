from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="MID_")

    data_dir: Path = Path("data")
    db_url: str | None = None          # derived if unset
    default_imgsz: int = 640
    default_conf: float = 0.25
    device: str = "auto"               # auto | cpu | mps | cuda:0

    @property
    def images_dir(self) -> Path:  return self.data_dir / "images"
    @property
    def datasets_dir(self) -> Path: return self.data_dir / "datasets"
    @property
    def runs_dir(self) -> Path:    return self.data_dir / "runs"
    @property
    def models_dir(self) -> Path:  return self.data_dir / "models"
    @property
    def pretrained_dir(self) -> Path: return self.data_dir / "pretrained"
    @property
    def faces_dir(self) -> Path:   return self.data_dir / "faces"

    def resolved_db_url(self) -> str:
        return self.db_url or f"sqlite:///{(self.data_dir / 'app.db').resolve()}"

    def ensure_dirs(self) -> None:
        for d in (self.images_dir, self.datasets_dir, self.runs_dir, self.models_dir,
                  self.pretrained_dir, self.faces_dir):
            d.mkdir(parents=True, exist_ok=True)


settings = Settings()
