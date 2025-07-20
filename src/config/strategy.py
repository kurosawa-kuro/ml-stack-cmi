"""
CMI Competition Project Strategy Configuration
Dynamic strategy loading based on project phase and environment variables
"""

import os
import logging
from enum import Enum
from typing import List, Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class ProjectPhase(Enum):
    """Project development phases"""
    BASELINE = "baseline"
    OPTIMIZATION = "optimization"
    ENSEMBLE = "ensemble"


class AlgorithmStrategy:
    """Algorithm strategy configuration"""
    
    def __init__(self, phase: ProjectPhase):
        self.phase = phase
        self._configure_algorithms()
    
    def _configure_algorithms(self):
        """Configure algorithms based on current phase"""
        if self.phase == ProjectPhase.BASELINE:
            self.enabled_algorithms = ["lightgbm"]
            self.primary_algorithm = "lightgbm"
            self.ensemble_enabled = False
            self.focus = "single_algorithm_mastery"
            
        elif self.phase == ProjectPhase.OPTIMIZATION:
            self.enabled_algorithms = ["lightgbm", "xgboost"]
            self.primary_algorithm = "lightgbm"
            self.ensemble_enabled = False
            self.focus = "hyperparameter_tuning"
            
        elif self.phase == ProjectPhase.ENSEMBLE:
            self.enabled_algorithms = ["lightgbm", "xgboost", "catboost", "cnn"]
            self.primary_algorithm = "lightgbm"
            self.ensemble_enabled = True
            self.focus = "model_diversity"
        
        # Override with environment variables if provided
        env_primary = os.getenv("PRIMARY_ALGORITHM")
        if env_primary:
            self.primary_algorithm = env_primary
            
        env_ensemble = os.getenv("ENSEMBLE_ENABLED", "").lower()
        if env_ensemble in ["true", "1", "yes"]:
            self.ensemble_enabled = True
        elif env_ensemble in ["false", "0", "no"]:
            self.ensemble_enabled = False


class TargetConfiguration:
    """Target score configuration by phase"""
    
    def __init__(self, phase: ProjectPhase):
        self.phase = phase
        self._configure_targets()
    
    def _configure_targets(self):
        """Configure target scores based on phase"""
        targets = {
            ProjectPhase.BASELINE: {
                "cv_score": 0.50,
                "lb_score": 0.50,
                "percentile": 70,
                "description": "Baseline establishment"
            },
            ProjectPhase.OPTIMIZATION: {
                "cv_score": 0.57,
                "lb_score": 0.56,
                "percentile": 65,
                "description": "Feature engineering completion"
            },
            ProjectPhase.ENSEMBLE: {
                "cv_score": 0.62,
                "lb_score": 0.60,
                "percentile": 60,
                "description": "Bronze medal achievement"
            }
        }
        
        config = targets[self.phase]
        self.cv_score = float(os.getenv("TARGET_CV_SCORE", config["cv_score"]))
        self.lb_score = float(os.getenv("TARGET_LB_SCORE", config["lb_score"]))
        self.bronze_score = float(os.getenv("TARGET_BRONZE_SCORE", 0.60))
        self.percentile = config["percentile"]
        self.description = config["description"]


class DataConfiguration:
    """Data processing configuration"""
    
    def __init__(self):
        self.source_type = os.getenv("DATA_SOURCE_TYPE", "duckdb")
        self.source_path = os.getenv("DATA_SOURCE_PATH", "data/kaggle_datasets.duckdb")
        
        # Cross-validation settings
        self.cv_strategy = os.getenv("CV_STRATEGY", "groupkfold")
        self.cv_folds = int(os.getenv("CV_FOLDS", "5"))
        self.group_column = os.getenv("GROUP_COLUMN", "participant_id")
        
        # Feature engineering settings
        self.tsfresh_enabled = os.getenv("TSFRESH_ENABLED", "true").lower() == "true"
        self.fft_enabled = os.getenv("FFT_ENABLED", "true").lower() == "true"
        self.multimodal_fusion = os.getenv("MULTIMODAL_FUSION", "true").lower() == "true"
        self.max_features = int(os.getenv("MAX_FEATURES", "100"))


class ResourceConfiguration:
    """Computational resource configuration"""
    
    def __init__(self):
        self.max_memory_gb = int(os.getenv("MAX_MEMORY_GB", "16"))
        self.max_cpu_cores = int(os.getenv("MAX_CPU_CORES", "8"))
        self.parallel_jobs = int(os.getenv("PARALLEL_JOBS", "4"))
        self.gpu_enabled = os.getenv("GPU_ENABLED", "false").lower() == "true"
        self.gpu_device_id = int(os.getenv("GPU_DEVICE_ID", "0"))


class ProjectConfig:
    """Main project configuration class"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize project configuration
        
        Args:
            config_path: Path to YAML config file (optional)
        """
        # Load environment variables first
        self._load_env_config()
        
        # Determine current phase
        phase_str = os.getenv("PROJECT_PHASE", "baseline").lower()
        try:
            self.phase = ProjectPhase(phase_str)
        except ValueError:
            logger.warning(f"Invalid PROJECT_PHASE '{phase_str}', defaulting to baseline")
            self.phase = ProjectPhase.BASELINE
        
        # Initialize strategy components
        self.algorithm_strategy = AlgorithmStrategy(self.phase)
        self.targets = TargetConfiguration(self.phase)
        self.data = DataConfiguration()
        self.resources = ResourceConfiguration()
        
        # Load YAML config if provided
        if config_path:
            self._load_yaml_config(config_path)
        
        logger.info(f"Project configured for {self.phase.value} phase")
        logger.info(f"Primary algorithm: {self.algorithm_strategy.primary_algorithm}")
        logger.info(f"Enabled algorithms: {self.algorithm_strategy.enabled_algorithms}")
        logger.info(f"Target CV score: {self.targets.cv_score}")
    
    def _load_env_config(self):
        """Load environment variables from .env file if exists"""
        env_path = Path(".env")
        if env_path.exists():
            try:
                with open(env_path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#'):
                            if '=' in line:
                                key, value = line.split('=', 1)
                                os.environ.setdefault(key.strip(), value.strip())
            except Exception as e:
                logger.warning(f"Error loading .env file: {e}")
    
    def _load_yaml_config(self, config_path: str):
        """Load additional configuration from YAML file"""
        try:
            import yaml
            with open(config_path, 'r') as f:
                yaml_config = yaml.safe_load(f)
            
            # Override settings with YAML values
            # This would merge YAML config with environment variables
            logger.info(f"Loaded additional config from {config_path}")
            
        except ImportError:
            logger.warning("PyYAML not installed, skipping YAML config loading")
        except Exception as e:
            logger.warning(f"Error loading YAML config: {e}")
    
    def get_enabled_algorithms(self) -> List[str]:
        """Get list of enabled algorithms for current phase"""
        return self.algorithm_strategy.enabled_algorithms
    
    def is_algorithm_enabled(self, algorithm: str) -> bool:
        """Check if specific algorithm is enabled"""
        return algorithm in self.algorithm_strategy.enabled_algorithms
    
    def should_use_ensemble(self) -> bool:
        """Check if ensemble methods should be used"""
        return self.algorithm_strategy.ensemble_enabled
    
    def get_model_params(self, algorithm: str) -> Dict[str, Any]:
        """Get model-specific parameters"""
        if algorithm == "lightgbm":
            return {
                "objective": os.getenv("LIGHTGBM_OBJECTIVE", "multiclass"),
                "num_class": int(os.getenv("LIGHTGBM_NUM_CLASS", "18")),
                "learning_rate": float(os.getenv("LIGHTGBM_LEARNING_RATE", "0.1")),
                "num_leaves": int(os.getenv("LIGHTGBM_NUM_LEAVES", "31")),
                "random_state": int(os.getenv("LIGHTGBM_RANDOM_STATE", "42")),
                "verbose": -1
            }
        elif algorithm == "xgboost":
            return {
                "learning_rate": float(os.getenv("XGBOOST_LEARNING_RATE", "0.1")),
                "max_depth": int(os.getenv("XGBOOST_MAX_DEPTH", "6")),
                "n_estimators": int(os.getenv("XGBOOST_N_ESTIMATORS", "100")),
                "random_state": 42
            }
        else:
            return {}
    
    def get_phase_description(self) -> str:
        """Get human-readable description of current phase"""
        descriptions = {
            ProjectPhase.BASELINE: "Single algorithm mastery with LightGBM",
            ProjectPhase.OPTIMIZATION: "Hyperparameter tuning and feature optimization",
            ProjectPhase.ENSEMBLE: "Model diversity and ensemble methods"
        }
        return descriptions.get(self.phase, "Unknown phase")
    
    def validate_configuration(self) -> List[str]:
        """Validate current configuration and return any warnings"""
        warnings = []
        
        # Check phase consistency
        if self.phase == ProjectPhase.BASELINE and len(self.algorithm_strategy.enabled_algorithms) > 1:
            warnings.append("Baseline phase should focus on single algorithm")
        
        # Check resource limits
        if self.resources.max_memory_gb < 8:
            warnings.append("Memory limit may be insufficient for tsfresh processing")
        
        # Check CV configuration
        if self.data.cv_strategy != "groupkfold":
            warnings.append("GroupKFold strongly recommended for CMI competition")
        
        return warnings


# Global configuration instance
_project_config: Optional[ProjectConfig] = None


def get_project_config(config_path: Optional[str] = None, reload: bool = False) -> ProjectConfig:
    """
    Get global project configuration instance
    
    Args:
        config_path: Path to YAML config file
        reload: Force reload of configuration
        
    Returns:
        ProjectConfig instance
    """
    global _project_config
    
    if _project_config is None or reload:
        _project_config = ProjectConfig(config_path)
    
    return _project_config


def set_project_phase(phase: ProjectPhase) -> None:
    """
    Change project phase and reload configuration
    
    Args:
        phase: New project phase
    """
    os.environ["PROJECT_PHASE"] = phase.value
    get_project_config(reload=True)
    logger.info(f"Project phase changed to: {phase.value}")


# Convenience functions for common checks
def is_baseline_phase() -> bool:
    """Check if currently in baseline phase"""
    return get_project_config().phase == ProjectPhase.BASELINE


def is_ensemble_enabled() -> bool:
    """Check if ensemble methods are enabled"""
    return get_project_config().should_use_ensemble()


def get_primary_algorithm() -> str:
    """Get primary algorithm for current phase"""
    return get_project_config().algorithm_strategy.primary_algorithm


def should_use_single_algorithm() -> bool:
    """Check if should use single algorithm (baseline phase logic)"""
    config = get_project_config()
    return config.phase == ProjectPhase.BASELINE or len(config.get_enabled_algorithms()) == 1