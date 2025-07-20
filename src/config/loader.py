"""
Configuration loading utilities
Handles YAML and environment variable loading
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


def load_env_config(env_file: str = ".env") -> Dict[str, str]:
    """
    Load environment variables from .env file
    
    Args:
        env_file: Path to .env file
        
    Returns:
        Dictionary of environment variables
    """
    env_vars = {}
    env_path = Path(env_file)
    
    if not env_path.exists():
        logger.warning(f"Environment file {env_file} not found")
        return env_vars
    
    try:
        with open(env_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                
                # Skip empty lines and comments
                if not line or line.startswith('#'):
                    continue
                
                # Parse key=value pairs
                if '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    
                    # Remove quotes if present
                    if value.startswith('"') and value.endswith('"'):
                        value = value[1:-1]
                    elif value.startswith("'") and value.endswith("'"):
                        value = value[1:-1]
                    
                    env_vars[key] = value
                    
                    # Set in os.environ if not already set
                    os.environ.setdefault(key, value)
                else:
                    logger.warning(f"Invalid line {line_num} in {env_file}: {line}")
    
    except Exception as e:
        logger.error(f"Error loading {env_file}: {e}")
    
    logger.info(f"Loaded {len(env_vars)} environment variables from {env_file}")
    return env_vars


def load_config(config_file: str = "config/project_config.yaml") -> Optional[Dict[str, Any]]:
    """
    Load configuration from YAML file
    
    Args:
        config_file: Path to YAML configuration file
        
    Returns:
        Configuration dictionary or None if loading fails
    """
    config_path = Path(config_file)
    
    if not config_path.exists():
        logger.warning(f"Configuration file {config_file} not found")
        return None
    
    try:
        import yaml
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        logger.info(f"Loaded configuration from {config_file}")
        return config
        
    except ImportError:
        logger.error("PyYAML not installed. Install with: pip install pyyaml")
        return None
    
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file {config_file}: {e}")
        return None
    
    except Exception as e:
        logger.error(f"Error loading configuration from {config_file}: {e}")
        return None


def get_config_value(key: str, default: Any = None, config: Optional[Dict[str, Any]] = None) -> Any:
    """
    Get configuration value with fallback priority:
    1. Environment variable
    2. YAML config
    3. Default value
    
    Args:
        key: Configuration key (supports dot notation for nested keys)
        default: Default value if key not found
        config: YAML configuration dictionary
        
    Returns:
        Configuration value
    """
    # First check environment variables (highest priority)
    env_key = key.upper().replace('.', '_')
    env_value = os.getenv(env_key)
    if env_value is not None:
        return _convert_env_value(env_value)
    
    # Then check YAML config
    if config is not None:
        try:
            value = config
            for part in key.split('.'):
                value = value[part]
            return value
        except (KeyError, TypeError):
            pass
    
    # Return default
    return default


def _convert_env_value(value: str) -> Any:
    """
    Convert environment variable string to appropriate type
    
    Args:
        value: String value from environment variable
        
    Returns:
        Converted value (bool, int, float, or string)
    """
    # Boolean conversion
    if value.lower() in ('true', 'yes', '1', 'on'):
        return True
    elif value.lower() in ('false', 'no', '0', 'off'):
        return False
    
    # Numeric conversion
    try:
        # Try integer first
        if '.' not in value:
            return int(value)
        else:
            return float(value)
    except ValueError:
        pass
    
    # Return as string
    return value


def validate_config_file(config_file: str = "config/project_config.yaml") -> bool:
    """
    Validate configuration file structure
    
    Args:
        config_file: Path to configuration file
        
    Returns:
        True if valid, False otherwise
    """
    config = load_config(config_file)
    if config is None:
        return False
    
    # Check required sections
    required_sections = ['project', 'competition', 'training', 'data']
    missing_sections = []
    
    for section in required_sections:
        if section not in config:
            missing_sections.append(section)
    
    if missing_sections:
        logger.error(f"Missing required configuration sections: {missing_sections}")
        return False
    
    # Validate project section
    project_config = config.get('project', {})
    if 'phase' not in project_config:
        logger.error("Missing 'phase' in project configuration")
        return False
    
    valid_phases = ['baseline', 'optimization', ensemble']
    if project_config['phase'] not in valid_phases:
        logger.error(f"Invalid phase '{project_config['phase']}'. Must be one of: {valid_phases}")
        return False
    
    logger.info("Configuration file validation passed")
    return True


def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge multiple configuration dictionaries
    Later configs override earlier ones
    
    Args:
        *configs: Configuration dictionaries to merge
        
    Returns:
        Merged configuration dictionary
    """
    merged = {}
    
    for config in configs:
        if config is not None:
            merged = _deep_merge(merged, config)
    
    return merged


def _deep_merge(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge two dictionaries
    
    Args:
        dict1: Base dictionary
        dict2: Dictionary to merge (takes precedence)
        
    Returns:
        Merged dictionary
    """
    result = dict1.copy()
    
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    
    return result