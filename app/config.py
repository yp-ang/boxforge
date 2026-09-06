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

    def data_path(self, value: str) -> Path:
        """Resolve a path stored in the DB (step 09 §3). New rows store paths relative
        to data_dir, so the same row resolves correctly whether this process is conda
        on a Mac or the Docker container mounting data/ at /app/data — the value that
        differs between environments is data_dir, not the suffix under it. A legacy
        absolute path (written before this existed) is honoured as-is; it only breaks
        if you move data/ to a different absolute location without going through the
        bind mount, which is the one thing this whole scheme exists to avoid."""
        p = Path(value)
        return p if p.is_absolute() else (self.data_dir / p)

    def rel_data_path(self, path: Path) -> str:
        """Inverse of data_path() — call this right before storing a path in the DB."""
        return str(Path(path).resolve().relative_to(self.data_dir.resolve()))


settings = Settings()
