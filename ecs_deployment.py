#!/usr/bin/env python3
"""
ECS Deployment Script for Interview Analysis API
Creates all necessary AWS resources for ECS deployment
"""

import boto3
import json
import os
import time
import subprocess
from pathlib import Path

class ECSDeployer:
    def __init__(self):
        self.region = os.environ.get('AWS_DEFAULT_REGION', 'us-east-1')
        self.account_id = boto3.client('sts').get_caller_identity()['Account']
        
        # Initialize AWS clients
        self.ecr = boto3.client('ecr', region_name=self.region)
        self.ecs = boto3.client('ecs', region_name=self.region)
        self.iam = boto3.client('iam', region_name=self.region)
        self.ec2 = boto3.client('ec2', region_name=self.region)
        self.elbv2 = boto3.client('elbv2', region_name=self.region)
        self.logs = boto3.client('logs', region_name=self.region)
        
        # Configuration
        self.app_name = "interview-analysis"
        self.cluster_name = f"{self.app_name}-cluster"
        self.service_name = f"{self.app_name}-service"
        self.task_family = f"{self.app_name}-task"
        self.repository_name = f"{self.app_name}-repo"
        
    def create_ecr_repository(self):
        """Create ECR repository for the Docker image"""
        print("🏗️ Creating ECR repository...")
        
        try:
            response = self.ecr.create_repository(
                repositoryName=self.repository_name,
                imageScanningConfiguration={
                    'scanOnPush': True
                }
            )
            print(f"✅ ECR repository created: {response['repository']['repositoryUri']}")
            return response['repository']['repositoryUri']
        except self.ecr.exceptions.RepositoryAlreadyExistsException:
            print(f"✅ ECR repository already exists: {self.repository_name}")
            return f"{self.account_id}.dkr.ecr.{self.region}.amazonaws.com/{self.repository_name}"
    
    def build_and_push_image(self, repository_uri):
        """Build and push Docker image to ECR"""
        print("🐳 Building and pushing Docker image...")
        
        # Get ECR login token
        auth_response = self.ecr.get_authorization_token()
        username, password = base64.b64decode(auth_response['authorizationData'][0]['authorizationToken']).decode().split(':')
        registry = auth_response['authorizationData'][0]['proxyEndpoint']
        
        # Login to ECR
        subprocess.run([
            "docker", "login", "-u", username, "-p", password, registry
        ], check=True)
        
        # Build image
        image_tag = f"{repository_uri}:latest"
        subprocess.run([
            "docker", "build", "-f", "Dockerfile.ecs", "-t", image_tag, "."
        ], check=True)
        
        # Push image
        subprocess.run(["docker", "push", image_tag], check=True)
        
        print(f"✅ Image pushed successfully: {image_tag}")
        return image_tag
    
    def create_iam_role(self, role_name, policy_document):
        """Create IAM role with specified policy"""
        try:
            # Create role
            self.iam.create_role(
                RoleName=role_name,
                AssumeRolePolicyDocument=json.dumps(policy_document)
            )
            
            # Attach policy
            self.iam.put_role_policy(
                RoleName=role_name,
                PolicyName=f"{role_name}-policy",
                PolicyDocument=json.dumps(policy_document)
            )
            
            print(f"✅ IAM role created: {role_name}")
            return f"arn:aws:iam::{self.account_id}:role/{role_name}"
            
        except self.iam.exceptions.EntityAlreadyExistsException:
            print(f"✅ IAM role already exists: {role_name}")
            return f"arn:aws:iam::{self.account_id}:role/{role_name}"
    
    def create_task_execution_role(self):
        """Create ECS task execution role"""
        assume_role_policy = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Principal": {"Service": "ecs-tasks.amazonaws.com"},
                    "Action": "sts:AssumeRole"
                }
            ]
        }
        
        task_execution_policy = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Action": [
                        "ecr:GetAuthorizationToken",
                        "ecr:BatchCheckLayerAvailability",
                        "ecr:GetDownloadUrlForLayer",
                        "ecr:BatchGetImage",
                        "logs:CreateLogStream",
                        "logs:PutLogEvents"
                    ],
                    "Resource": "*"
                }
            ]
        }
        
        role_arn = self.create_iam_role(f"{self.app_name}-task-execution-role", assume_role_policy)
        
        # Attach AWS managed policy for ECS task execution
        self.iam.attach_role_policy(
            RoleName=f"{self.app_name}-task-execution-role",
            PolicyArn="arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
        )
        
        return role_arn
    
    def create_task_role(self):
        """Create ECS task role for application permissions"""
        assume_role_policy = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Principal": {"Service": "ecs-tasks.amazonaws.com"},
                    "Action": "sts:AssumeRole"
                }
            ]
        }
        
        task_policy = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Action": [
                        "logs:CreateLogStream",
                        "logs:PutLogEvents"
                    ],
                    "Resource": "*"
                }
            ]
        }
        
        return self.create_iam_role(f"{self.app_name}-task-role", assume_role_policy)
    
    def create_log_group(self):
        """Create CloudWatch log group"""
        print("📝 Creating CloudWatch log group...")
        
        log_group_name = f"/ecs/{self.task_family}"
        
        try:
            self.logs.create_log_group(logGroupName=log_group_name)
            print(f"✅ Log group created: {log_group_name}")
        except self.logs.exceptions.ResourceAlreadyExistsException:
            print(f"✅ Log group already exists: {log_group_name}")
        
        return log_group_name
    
    def create_vpc_and_security_group(self):
        """Create VPC and security group for ECS"""
        print("🌐 Creating VPC and security group...")
        
        # Create VPC
        vpc_response = self.ec2.create_vpc(
            CidrBlock='10.0.0.0/16',
            EnableDnsHostnames=True,
            EnableDnsSupport=True
        )
        vpc_id = vpc_response['Vpc']['VpcId']
        
        # Create Internet Gateway
        igw_response = self.ec2.create_internet_gateway()
        igw_id = igw_response['InternetGateway']['InternetGatewayId']
        
        # Attach Internet Gateway to VPC
        self.ec2.attach_internet_gateway(
            InternetGatewayId=igw_id,
            VpcId=vpc_id
        )
        
        # Create public subnet
        subnet_response = self.ec2.create_subnet(
            VpcId=vpc_id,
            CidrBlock='10.0.1.0/24',
            AvailabilityZone=f"{self.region}a"
        )
        subnet_id = subnet_response['Subnet']['SubnetId']
        
        # Create route table
        route_table_response = self.ec2.create_route_table(VpcId=vpc_id)
        route_table_id = route_table_response['RouteTable']['RouteTableId']
        
        # Add route to internet gateway
        self.ec2.create_route(
            RouteTableId=route_table_id,
            DestinationCidrBlock='0.0.0.0/0',
            GatewayId=igw_id
        )
        
        # Associate route table with subnet
        self.ec2.associate_route_table(
            RouteTableId=route_table_id,
            SubnetId=subnet_id
        )
        
        # Create security group
        sg_response = self.ec2.create_security_group(
            GroupName=f"{self.app_name}-sg",
            Description=f"Security group for {self.app_name}",
            VpcId=vpc_id
        )
        sg_id = sg_response['GroupId']
        
        # Add inbound rule for HTTP
        self.ec2.authorize_security_group_ingress(
            GroupId=sg_id,
            IpPermissions=[
                {
                    'IpProtocol': 'tcp',
                    'FromPort': 8000,
                    'ToPort': 8000,
                    'IpRanges': [{'CidrIp': '0.0.0.0/0'}]
                }
            ]
        )
        
        print(f"✅ VPC created: {vpc_id}")
        print(f"✅ Security group created: {sg_id}")
        
        return vpc_id, subnet_id, sg_id
    
    def create_application_load_balancer(self, vpc_id, subnet_id, sg_id):
        """Create Application Load Balancer"""
        print("⚖️ Creating Application Load Balancer...")
        
        # Create ALB
        alb_response = self.elbv2.create_load_balancer(
            Name=f"{self.app_name}-alb",
            Subnets=[subnet_id],
            SecurityGroups=[sg_id],
            Scheme='internet-facing',
            Type='application'
        )
        
        alb_arn = alb_response['LoadBalancers'][0]['LoadBalancerArn']
        alb_dns = alb_response['LoadBalancers'][0]['DNSName']
        
        # Create target group
        tg_response = self.elbv2.create_target_group(
            Name=f"{self.app_name}-tg",
            Protocol='HTTP',
            Port=8000,
            VpcId=vpc_id,
            TargetType='ip',
            HealthCheckProtocol='HTTP',
            HealthCheckPath='/health',
            HealthCheckIntervalSeconds=30,
            HealthCheckTimeoutSeconds=5,
            HealthyThresholdCount=2,
            UnhealthyThresholdCount=2
        )
        
        tg_arn = tg_response['TargetGroups'][0]['TargetGroupArn']
        
        # Create listener
        self.elbv2.create_listener(
            LoadBalancerArn=alb_arn,
            Protocol='HTTP',
            Port=80,
            DefaultActions=[
                {
                    'Type': 'forward',
                    'TargetGroupArn': tg_arn
                }
            ]
        )
        
        print(f"✅ ALB created: {alb_dns}")
        print(f"✅ Target group created: {tg_arn}")
        
        return alb_arn, tg_arn, alb_dns
    
    def create_ecs_cluster(self):
        """Create ECS cluster"""
        print("🏗️ Creating ECS cluster...")
        
        try:
            response = self.ecs.create_cluster(
                clusterName=self.cluster_name,
                capacityProviders=['FARGATE'],
                defaultCapacityProviderStrategy=[
                    {
                        'capacityProvider': 'FARGATE',
                        'weight': 1
                    }
                ]
            )
            print(f"✅ ECS cluster created: {response['cluster']['clusterArn']}")
        except self.ecs.exceptions.ClusterAlreadyExistsException:
            print(f"✅ ECS cluster already exists: {self.cluster_name}")
    
    def create_task_definition(self, image_uri, task_execution_role_arn, task_role_arn, log_group_name):
        """Create ECS task definition"""
        print("📋 Creating ECS task definition...")
        
        task_definition = {
            'family': self.task_family,
            'networkMode': 'awsvpc',
            'requiresCompatibilities': ['FARGATE'],
            'cpu': '1024',
            'memory': '2048',
            'executionRoleArn': task_execution_role_arn,
            'taskRoleArn': task_role_arn,
            'containerDefinitions': [
                {
                    'name': f"{self.app_name}-container",
                    'image': image_uri,
                    'portMappings': [
                        {
                            'containerPort': 8000,
                            'protocol': 'tcp'
                        }
                    ],
                    'essential': True,
                    'logConfiguration': {
                        'logDriver': 'awslogs',
                        'options': {
                            'awslogs-group': log_group_name,
                            'awslogs-region': self.region,
                            'awslogs-stream-prefix': 'ecs'
                        }
                    },
                    'healthCheck': {
                        'command': ['CMD-SHELL', 'curl -f http://localhost:8000/health || exit 1'],
                        'interval': 30,
                        'timeout': 5,
                        'retries': 3,
                        'startPeriod': 60
                    }
                }
            ]
        }
        
        response = self.ecs.register_task_definition(**task_definition)
        print(f"✅ Task definition created: {response['taskDefinition']['taskDefinitionArn']}")
        return response['taskDefinition']['taskDefinitionArn']
    
    def create_ecs_service(self, task_definition_arn, tg_arn, subnet_id, sg_id):
        """Create ECS service"""
        print("🚀 Creating ECS service...")
        
        service_response = self.ecs.create_service(
            cluster=self.cluster_name,
            serviceName=self.service_name,
            taskDefinition=task_definition_arn,
            desiredCount=1,
            launchType='FARGATE',
            networkConfiguration={
                'awsvpcConfiguration': {
                    'subnets': [subnet_id],
                    'securityGroups': [sg_id],
                    'assignPublicIp': 'ENABLED'
                }
            },
            loadBalancers=[
                {
                    'targetGroupArn': tg_arn,
                    'containerName': f"{self.app_name}-container",
                    'containerPort': 8000
                }
            ]
        )
        
        print(f"✅ ECS service created: {service_response['service']['serviceArn']}")
        return service_response['service']['serviceArn']
    
    def deploy(self):
        """Deploy the complete ECS infrastructure"""
        print("🚀 Deploying Interview Analysis API to ECS")
        print("=" * 60)
        
        try:
            # Step 1: Create ECR repository
            repository_uri = self.create_ecr_repository()
            
            # Step 2: Build and push Docker image
            image_uri = self.build_and_push_image(repository_uri)
            
            # Step 3: Create IAM roles
            task_execution_role_arn = self.create_task_execution_role()
            task_role_arn = self.create_task_role()
            
            # Step 4: Create CloudWatch log group
            log_group_name = self.create_log_group()
            
            # Step 5: Create VPC and security group
            vpc_id, subnet_id, sg_id = self.create_vpc_and_security_group()
            
            # Step 6: Create Application Load Balancer
            alb_arn, tg_arn, alb_dns = self.create_application_load_balancer(vpc_id, subnet_id, sg_id)
            
            # Step 7: Create ECS cluster
            self.create_ecs_cluster()
            
            # Step 8: Create task definition
            task_definition_arn = self.create_task_definition(
                image_uri, task_execution_role_arn, task_role_arn, log_group_name
            )
            
            # Step 9: Create ECS service
            service_arn = self.create_ecs_service(task_definition_arn, tg_arn, subnet_id, sg_id)
            
            print("\n🎉 Deployment completed successfully!")
            print(f"🌐 API Endpoint: http://{alb_dns}")
            print(f"📊 ECS Console: https://console.aws.amazon.com/ecs/home?region={self.region}#/clusters/{self.cluster_name}")
            print(f"📝 Logs: https://console.aws.amazon.com/cloudwatch/home?region={self.region}#logsV2:log-groups/log-group/{log_group_name.replace('/', '$252F')}")
            
            # Save configuration
            config = {
                "api_endpoint": f"http://{alb_dns}",
                "cluster_name": self.cluster_name,
                "service_name": self.service_name,
                "task_family": self.task_family,
                "repository_uri": repository_uri,
                "region": self.region
            }
            
            with open('ecs_config.json', 'w') as f:
                json.dump(config, f, indent=2)
            
            print(f"📁 Configuration saved to ecs_config.json")
            
            return config
            
        except Exception as e:
            print(f"❌ Deployment failed: {str(e)}")
            raise

def main():
    """Main function"""
    deployer = ECSDeployer()
    config = deployer.deploy()
    
    if config:
        print(f"\n🧪 Test your API:")
        print(f"curl http://{config['api_endpoint'].replace('http://', '')}/")
        print(f"curl http://{config['api_endpoint'].replace('http://', '')}/health")

if __name__ == "__main__":
    main() 