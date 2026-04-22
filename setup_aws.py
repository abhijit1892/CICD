import os
import configparser
from pathlib import Path

def main():
    print("="*50)
    print("AWS Credentials Setup (Bypassing broken AWS CLI)")
    print("="*50)
    
    access_key = input("Enter your AWS Access Key ID: ").strip()
    secret_key = input("Enter your AWS Secret Access Key: ").strip()
    region = input("Enter your AWS Region (e.g., us-east-1): ").strip()
    
    if not access_key or not secret_key:
        print("Please provide valid keys.")
        return
        
    aws_dir = Path.home() / ".aws"
    aws_dir.mkdir(exist_ok=True)
    
    # Write credentials file
    cred_file = aws_dir / "credentials"
    cred_config = configparser.ConfigParser()
    
    if cred_file.exists():
        cred_config.read(cred_file)
        
    if 'default' not in cred_config.sections():
        cred_config.add_section('default')
        
    cred_config['default']['aws_access_key_id'] = access_key
    cred_config['default']['aws_secret_access_key'] = secret_key
    
    with open(cred_file, 'w') as f:
        cred_config.write(f)
        
    # Write config file for region
    config_file = aws_dir / "config"
    conf_config = configparser.ConfigParser()
    
    if config_file.exists():
        conf_config.read(config_file)
        
    if 'default' not in conf_config.sections():
        conf_config.add_section('default')
        
    conf_config['default']['region'] = region
    
    with open(config_file, 'w') as f:
        conf_config.write(f)
        
    print("\n✅ Successfully configured ~/.aws/credentials!")
    print("You can now run: python3 run_training_job.py")

if __name__ == "__main__":
    main()
