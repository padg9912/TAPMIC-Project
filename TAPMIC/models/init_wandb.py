#!/usr/bin/env python3
# init_wandb.py
# Script to initialize Weights & Biases for TAPMIC project

import os
import sys
import argparse
import logging
import wandb

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

def init_wandb(api_key=None):
    """
    Initialize Weights & Biases with the given API key
    
    Args:
        api_key: Optional API key (if not provided, will prompt or use existing)
    """
    try:
        if api_key:
            # Login with provided API key
            wandb.login(key=api_key)
            logging.info("Successfully logged in to Weights & Biases with provided API key")
        else:
            # Try to use existing credentials or prompt for login
            wandb.login()
            logging.info("Successfully logged in to Weights & Biases with existing credentials")
        
        # Test connection with a simple initialization
        test_run = wandb.init(project="TAPMIC", name="test_connection", settings=wandb.Settings(silent=True))
        wandb.log({"test": 1.0})
        test_run.finish()
        
        logging.info("Weights & Biases initialization test successful")
        return True
    except Exception as e:
        logging.error(f"Error initializing Weights & Biases: {e}")
        return False

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Initialize Weights & Biases for TAPMIC")
    parser.add_argument("--api_key", type=str, help="Weights & Biases API key")
    return parser.parse_args()

def main():
    """Main function"""
    args = parse_args()
    success = init_wandb(args.api_key)
    
    if success:
        logging.info("Weights & Biases setup completed successfully")
        if os.path.exists(os.path.expanduser("~/.netrc")):
            logging.info("API key stored in ~/.netrc for future use")
        
        # Set up environment
        config_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "configs")
        os.makedirs(config_dir, exist_ok=True)
        
        # Print instructions
        print("\n" + "="*50)
        print("Weights & Biases Setup Complete")
        print("="*50)
        print("\nTo use wandb with TAPMIC, run your training with:")
        print("  python run_roberta.py --wandb --run_name your_run_name [other options]")
        print("\nTo disable wandb for a run, simply omit the --wandb flag")
        print("\nYour runs will be available at: https://wandb.ai/your-username/TAPMIC")
        print("="*50 + "\n")
        
        return 0
    else:
        logging.error("Failed to set up Weights & Biases")
        print("\n" + "="*50)
        print("Weights & Biases Setup Failed")
        print("="*50)
        print("\nTo retry setup, run:")
        print("  python init_wandb.py --api_key YOUR_API_KEY")
        print("\nYou can get your API key from: https://wandb.ai/settings")
        print("="*50 + "\n")
        
        return 1

if __name__ == "__main__":
    sys.exit(main()) 