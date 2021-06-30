from .config_loader import ConfigLoader


Config = ConfigLoader(path="config/app_config.json")
Config.load()
