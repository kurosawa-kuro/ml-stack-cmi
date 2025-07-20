"""
Configuration management for CMI Competition Project
Provides configuration-driven development capabilities
"""

from .strategy import ProjectConfig, ProjectPhase, get_project_config
from .loader import load_config, load_env_config

__all__ = ['ProjectConfig', 'ProjectPhase', 'get_project_config', 'load_config', 'load_env_config']