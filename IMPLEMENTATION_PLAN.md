# AI Video Codec Framework - Technical Implementation Plan

## Overview

This document provides detailed technical specifications for implementing the autonomous AI video codec development framework.

---

## 1. Project Structure

```
AiV1/
├── README.md
├── requirements.txt
├── setup.py
├── config/
│   ├── aws_config.yaml
│   ├── training_config.yaml
│   └── codec_config.yaml
├── orchestrator/
│   ├── __init__.py
│   ├── master.py              # Main orchestrator
│   ├── experiment_planner.py  # Experiment strategy
│   ├── cost_tracker.py        # AWS cost monitoring
│   └── reporter.py            # Hourly reporting
├── workers/
│   ├── __init__.py
│   ├── base_worker.py
│   ├── training_worker.py
│   ├── inference_worker.py
│   └── evaluation_worker.py
├── codec/
│   ├── __init__.py
│   ├── models/
│   │   ├── autoencoder.py
│   │   ├── hyperprior.py
│   │   ├── vqvae.py
│   │   ├── hybrid.py
│   │   └── generative.py
│   ├── encoder.py
│   ├── decoder.py
│   └── utils.py
├── training/
│   ├── __init__.py
│   ├── trainer.py
│   ├── losses.py
│   ├── dataset.py
│   └── augmentation.py
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py             # PSNR, SSIM, VMAF
│   ├── benchmarks.py          # Performance testing
│   └── visualizer.py
├── optimization/
│   ├── __init__.py
│   ├── quantization.py
│   ├── pruning.py
│   └── distillation.py
├── infrastructure/
│   ├── __init__.py
│   ├── aws_setup.py
│   ├── cloudformation/
│   │   ├── compute.yaml
│   │   ├── storage.yaml
│   │   └── networking.yaml
│   └── terraform/             # Alternative to CFN
│       ├── main.tf
│       ├── variables.tf
│       └── outputs.tf
├── scripts/
│   ├── deploy_framework.sh
│   ├── run_experiment.py
│   ├── evaluate_codec.py
│   └── optimize_model.py
└── tests/
    ├── test_codec.py
    ├── test_training.py
    └── test_evaluation.py
```

---

## 2. Core Components Implementation

### 2.1 Orchestrator Service

**File: `orchestrator/master.py`**

```python
class AutonomousOrchestrator:
    """
    Main orchestrator that manages the entire codec development lifecycle.
    Runs continuously until goals are achieved.
    """
    
    def __init__(self, config):
        self.config = config
        self.experiment_planner = ExperimentPlanner()
        self.cost_tracker = CostTracker()
        self.reporter = HourlyReporter()
        self.knowledge_base = ExperimentKnowledgeBase()
        
        # AWS clients
        self.sqs = boto3.client('sqs')
        self.dynamodb = boto3.resource('dynamodb')
        self.s3 = boto3.client('s3')
        
        # State
        self.experiments_completed = 0
        self.best_result = None
        self.goals_met = False
        
    def run(self):
        """Main orchestration loop"""
        while not self.goals_met:
            # Check budget
            if self.cost_tracker.monthly_cost > self.config.max_budget:
                self.scale_down()
                
            # Plan next experiment
            experiment = self.experiment_planner.next_experiment(
                self.knowledge_base
            )
            
            # Submit to training queue
            self.submit_experiment(experiment)
            
            # Monitor progress
            self.monitor_experiments()
            
            # Hourly reporting
            if self.should_report():
                self.reporter.generate_report()
                
            # Check if goals achieved
            self.goals_met = self.evaluate_goals()
            
            time.sleep(60)  # Check every minute
            
    def submit_experiment(self, experiment):
        """Submit experiment to training queue"""
        message = {
            'experiment_id': experiment.id,
            'architecture': experiment.architecture,
            'hyperparams': experiment.hyperparams,
            'timestamp': datetime.now().isoformat()
        }
        self.sqs.send_message(
            QueueUrl=self.config.training_queue_url,
            MessageBody=json.dumps(message)
        )
        
    def evaluate_goals(self):
        """Check if codec meets all requirements"""
        if self.best_result is None:
            return False
            
        return (
            self.best_result.bitrate_reduction >= 0.90 and
            self.best_result.psnr >= 95.0 and
            self.best_result.inference_fps >= 60.0
        )
```

**File: `orchestrator/experiment_planner.py`**

```python
class ExperimentPlanner:
    """
    Plans experiments using evolutionary algorithms and meta-learning.
    Learns from past experiments to guide future exploration.
    """
    
    def __init__(self):
        self.strategies = [
            'baseline_autoencoder',
            'scale_hyperprior',
            'vqvae_variant',
            'hybrid_semantic',
            'generative_refinement',
            'evolutionary_mutation'
        ]
        
    def next_experiment(self, knowledge_base):
        """
        Select next experiment based on:
        1. Past performance
        2. Unexplored areas
        3. Promising directions
        """
        
        # Stage-based strategy
        if knowledge_base.num_experiments < 5:
            # Early: Try diverse baselines
            strategy = self._select_baseline()
        elif knowledge_base.num_experiments < 20:
            # Mid: Optimize promising approaches
            strategy = self._select_optimization()
        else:
            # Late: Fine-tune best performers
            strategy = self._select_refinement()
            
        return self._generate_experiment(strategy, knowledge_base)
        
    def _select_baseline(self):
        """Select from proven baseline architectures"""
        return random.choice([
            'baseline_autoencoder',
            'scale_hyperprior',
            'vqvae_variant'
        ])
        
    def _select_optimization(self):
        """Select optimization strategy based on results"""
        # Use meta-learning model to predict promising direction
        return self._meta_learn_next_strategy()
        
    def _generate_experiment(self, strategy, knowledge_base):
        """Generate experiment configuration"""
        
        if strategy == 'evolutionary_mutation':
            # Mutate best performing model
            parent = knowledge_base.get_best_experiment()
            return self._mutate(parent)
            
        # Generate new configuration
        config = {
            'architecture': self._sample_architecture(strategy),
            'hyperparams': self._sample_hyperparams(),
            'loss_function': self._sample_loss(),
            'training_config': self._sample_training_config()
        }
        
        return Experiment(strategy, config)
```

**File: `orchestrator/cost_tracker.py`**

```python
class CostTracker:
    """
    Tracks AWS costs in real-time and enforces budget constraints.
    """
    
    def __init__(self, budget=5000):
        self.budget = budget
        self.ce_client = boto3.client('ce')  # Cost Explorer
        self.cloudwatch = boto3.client('cloudwatch')
        
    @property
    def monthly_cost(self):
        """Get current month-to-date cost"""
        start = datetime.now().replace(day=1).strftime('%Y-%m-%d')
        end = datetime.now().strftime('%Y-%m-%d')
        
        response = self.ce_client.get_cost_and_usage(
            TimePeriod={'Start': start, 'End': end},
            Granularity='MONTHLY',
            Metrics=['UnblendedCost']
        )
        
        total = float(
            response['ResultsByTime'][0]['Total']['UnblendedCost']['Amount']
        )
        return total
        
    def projected_monthly_cost(self):
        """Project cost for full month based on current usage"""
        days_elapsed = datetime.now().day
        days_in_month = calendar.monthrange(
            datetime.now().year, 
            datetime.now().month
        )[1]
        
        return self.monthly_cost * (days_in_month / days_elapsed)
        
    def check_budget_alert(self):
        """Check if approaching budget limits"""
        projected = self.projected_monthly_cost()
        
        if projected > self.budget * 0.95:
            return 'CRITICAL'
        elif projected > self.budget * 0.85:
            return 'HIGH'
        elif projected > self.budget * 0.70:
            return 'MEDIUM'
        return 'OK'
```

### 2.2 Training Worker

**File: `workers/training_worker.py`**

```python
class TrainingWorker:
    """
    Pulls experiments from queue and trains models.
    Runs on GPU instances (g5.4xlarge).
    """
    
    def __init__(self, worker_id):
        self.worker_id = worker_id
        self.sqs = boto3.client('sqs')
        self.s3 = boto3.client('s3')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def run(self):
        """Main worker loop"""
        while True:
            # Poll for messages
            messages = self.sqs.receive_message(
                QueueUrl=config.TRAINING_QUEUE_URL,
                MaxNumberOfMessages=1,
                WaitTimeSeconds=20
            )
            
            if 'Messages' not in messages:
                continue
                
            message = messages['Messages'][0]
            experiment = json.loads(message['Body'])
            
            try:
                # Train model
                result = self.train_experiment(experiment)
                
                # Save results
                self.save_results(experiment, result)
                
                # Delete message
                self.sqs.delete_message(
                    QueueUrl=config.TRAINING_QUEUE_URL,
                    ReceiptHandle=message['ReceiptHandle']
                )
                
            except Exception as e:
                logger.error(f"Training failed: {e}")
                # Message will return to queue
                
    def train_experiment(self, experiment):
        """Train a single experiment"""
        
        # Load data
        train_loader = self.load_training_data()
        
        # Build model
        model = self.build_model(experiment['architecture'])
        model = model.to(self.device)
        
        # Setup training
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        loss_fn = self.build_loss_function(experiment['loss_function'])
        
        # Training loop
        best_loss = float('inf')
        for epoch in range(experiment['num_epochs']):
            
            epoch_loss = self.train_epoch(
                model, train_loader, optimizer, loss_fn
            )
            
            # Report progress
            self.report_progress(experiment['experiment_id'], epoch, epoch_loss)
            
            # Save checkpoint
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                self.save_checkpoint(model, experiment, epoch)
                
        # Final evaluation
        metrics = self.evaluate_model(model)
        
        return {
            'experiment_id': experiment['experiment_id'],
            'final_loss': best_loss,
            'metrics': metrics,
            'model_path': self.save_final_model(model, experiment)
        }
        
    def train_epoch(self, model, loader, optimizer, loss_fn):
        """Train for one epoch"""
        model.train()
        total_loss = 0
        
        for batch in loader:
            frames = batch['frames'].to(self.device)
            
            optimizer.zero_grad()
            
            # Forward pass
            encoded = model.encode(frames)
            decoded = model.decode(encoded)
            
            # Compute loss
            loss = loss_fn(decoded, frames, encoded)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / len(loader)
```

### 2.3 Evaluation Worker

**File: `workers/evaluation_worker.py`**

```python
class EvaluationWorker:
    """
    Evaluates trained models against quality and performance metrics.
    """
    
    def __init__(self):
        self.metrics = {
            'psnr': PSNRMetric(),
            'ssim': SSIMMetric(),
            'vmaf': VMAFMetric(),
            'lpips': LPIPSMetric()
        }
        
    def evaluate(self, model_path, test_video_path):
        """Comprehensive evaluation of a codec model"""
        
        # Load model
        model = self.load_model(model_path)
        model.eval()
        
        # Load test video
        original_frames = self.load_video(test_video_path)
        
        # Encode
        start_time = time.time()
        bitstream = model.encode(original_frames)
        encode_time = time.time() - start_time
        
        # Decode
        start_time = time.time()
        decoded_frames = model.decode(bitstream)
        decode_time = time.time() - start_time
        
        # Quality metrics
        quality_metrics = {}
        for name, metric in self.metrics.items():
            quality_metrics[name] = metric.compute(
                original_frames, 
                decoded_frames
            )
            
        # Compression metrics
        bitrate = len(bitstream) * 8 / self.get_video_duration(test_video_path)
        original_bitrate = self.get_original_bitrate(test_video_path)
        compression_ratio = original_bitrate / bitrate
        
        # Performance metrics
        num_frames = len(original_frames)
        encode_fps = num_frames / encode_time
        decode_fps = num_frames / decode_time
        
        return {
            'quality': quality_metrics,
            'bitrate': bitrate,
            'compression_ratio': compression_ratio,
            'encode_fps': encode_fps,
            'decode_fps': decode_fps,
            'model_size': self.get_model_size(model)
        }
```

### 2.4 Codec Models

**File: `codec/models/hyperprior.py`**

```python
class ScaleHyperpriorCodec(nn.Module):
    """
    Neural compression with scale hyperprior (Ballé et al., 2018).
    Proven architecture for neural image/video compression.
    """
    
    def __init__(self, channels=128):
        super().__init__()
        
        # Analysis (Encoder)
        self.analysis = nn.Sequential(
            Conv3d(3, 128, kernel_size=5, stride=2, padding=2),
            GDN(128),
            Conv3d(128, 128, kernel_size=5, stride=2, padding=2),
            GDN(128),
            Conv3d(128, channels, kernel_size=5, stride=2, padding=2)
        )
        
        # Hyper analysis (encodes distribution parameters)
        self.hyper_analysis = nn.Sequential(
            Conv3d(channels, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            Conv3d(128, 128, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            Conv3d(128, 128, kernel_size=5, stride=2, padding=2)
        )
        
        # Hyper synthesis (decodes distribution parameters)
        self.hyper_synthesis = nn.Sequential(
            ConvTranspose3d(128, 128, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(),
            ConvTranspose3d(128, 128, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(),
            ConvTranspose3d(128, channels*2, kernel_size=3, stride=1, padding=1)
        )
        
        # Synthesis (Decoder)
        self.synthesis = nn.Sequential(
            ConvTranspose3d(channels, 128, kernel_size=5, stride=2, padding=2, output_padding=1),
            GDN(128, inverse=True),
            ConvTranspose3d(128, 128, kernel_size=5, stride=2, padding=2, output_padding=1),
            GDN(128, inverse=True),
            ConvTranspose3d(128, 3, kernel_size=5, stride=2, padding=2, output_padding=1)
        )
        
        # Entropy model
        self.entropy_bottleneck = EntropyBottleneck(128)
        self.gaussian_conditional = GaussianConditional(None)
        
    def forward(self, x):
        """Forward pass for training"""
        # Encode
        y = self.analysis(x)
        z = self.hyper_analysis(y)
        
        # Quantize (with noise during training)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        
        # Predict mean and scale for y
        params = self.hyper_synthesis(z_hat)
        means, scales = params.chunk(2, dim=1)
        
        # Quantize y with predicted distribution
        y_hat, y_likelihoods = self.gaussian_conditional(y, scales, means)
        
        # Decode
        x_hat = self.synthesis(y_hat)
        
        return {
            'x_hat': x_hat,
            'likelihoods': {'y': y_likelihoods, 'z': z_likelihoods}
        }
        
    def compress(self, x):
        """Compress to bitstream"""
        y = self.analysis(x)
        z = self.hyper_analysis(y)
        
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-3:])
        
        params = self.hyper_synthesis(z_hat)
        means, scales = params.chunk(2, dim=1)
        
        indexes = self.gaussian_conditional.build_indexes(scales)
        y_strings = self.gaussian_conditional.compress(y, indexes, means)
        
        return {'strings': [y_strings, z_strings], 'shape': z.size()[-3:]}
        
    def decompress(self, strings, shape):
        """Decompress from bitstream"""
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        
        params = self.hyper_synthesis(z_hat)
        means, scales = params.chunk(2, dim=1)
        
        indexes = self.gaussian_conditional.build_indexes(scales)
        y_hat = self.gaussian_conditional.decompress(strings[0], indexes, means)
        
        x_hat = self.synthesis(y_hat)
        return x_hat
```

**File: `codec/models/hybrid.py`**

```python
class HybridSemanticCodec(nn.Module):
    """
    Hybrid approach combining semantic extraction with neural compression.
    Best chance of achieving 90% compression with high quality.
    """
    
    def __init__(self):
        super().__init__()
        
        # Semantic extraction
        self.motion_estimator = OpticalFlowNet()
        self.scene_analyzer = SceneUnderstandingNet()
        
        # Keyframe codec (high quality)
        self.keyframe_codec = ScaleHyperpriorCodec(channels=256)
        
        # Inter-frame codec (motion + residuals)
        self.inter_codec = MotionCompensationCodec(channels=64)
        
        # Generative refinement
        self.refiner = GenerativeRefinementNet()
        
        self.keyframe_interval = 30  # Every 0.5s at 60fps
        
    def forward(self, frames):
        """
        frames: (B, T, C, H, W) - batch of video sequences
        """
        B, T, C, H, W = frames.shape
        
        # Identify keyframes
        keyframe_indices = list(range(0, T, self.keyframe_interval))
        
        reconstructed = []
        
        for i in range(T):
            if i in keyframe_indices:
                # Encode keyframe with high quality
                frame = frames[:, i]
                encoded = self.keyframe_codec(frame)
                reconstructed.append(encoded['x_hat'])
            else:
                # Encode as inter-frame
                ref_idx = (i // self.keyframe_interval) * self.keyframe_interval
                ref_frame = reconstructed[ref_idx]
                current_frame = frames[:, i]
                
                # Motion compensation
                motion = self.motion_estimator(ref_frame, current_frame)
                predicted = self.warp(ref_frame, motion)
                residual = current_frame - predicted
                
                # Encode residual
                encoded_residual = self.inter_codec(residual)
                
                # Reconstruct
                decoded = predicted + encoded_residual['x_hat']
                
                # Generative refinement
                refined = self.refiner(decoded, semantic_context)
                
                reconstructed.append(refined)
                
        return torch.stack(reconstructed, dim=1)
```

---

## 3. Infrastructure as Code

### 3.1 CloudFormation Template

**File: `infrastructure/cloudformation/compute.yaml`**

```yaml
AWSTemplateFormatVersion: '2010-09-09'
Description: 'AI Video Codec Framework - Compute Resources'

Parameters:
  ProjectName:
    Type: String
    Default: ai-video-codec
    
  Environment:
    Type: String
    Default: production
    AllowedValues:
      - development
      - production

Resources:
  # Orchestrator Instance
  OrchestratorInstance:
    Type: AWS::EC2::Instance
    Properties:
      InstanceType: c6i.xlarge
      ImageId: !Ref LatestAmiId
      IamInstanceProfile: !Ref OrchestratorInstanceProfile
      SecurityGroupIds:
        - !Ref OrchestratorSecurityGroup
      UserData:
        Fn::Base64: !Sub |
          #!/bin/bash
          # Install dependencies
          apt-get update
          apt-get install -y python3.10 python3-pip git
          
          # Clone repository
          cd /opt
          git clone https://github.com/your-org/ai-video-codec.git
          cd ai-video-codec
          
          # Install Python packages
          pip3 install -r requirements.txt
          
          # Start orchestrator
          python3 orchestrator/master.py --config config/aws_config.yaml
      Tags:
        - Key: Name
          Value: !Sub '${ProjectName}-orchestrator'
        - Key: Environment
          Value: !Ref Environment
          
  # Training Worker Auto Scaling Group
  TrainingWorkerLaunchTemplate:
    Type: AWS::EC2::LaunchTemplate
    Properties:
      LaunchTemplateName: !Sub '${ProjectName}-training-worker'
      LaunchTemplateData:
        InstanceType: g5.4xlarge
        ImageId: !Ref GpuAmiId
        IamInstanceProfile:
          Arn: !GetAtt TrainingWorkerInstanceProfile.Arn
        SecurityGroupIds:
          - !Ref WorkerSecurityGroup
        UserData:
          Fn::Base64: !Sub |
            #!/bin/bash
            # Setup GPU drivers
            /opt/setup_gpu.sh
            
            # Install dependencies
            pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118
            pip3 install -r /opt/ai-video-codec/requirements.txt
            
            # Start training worker
            python3 /opt/ai-video-codec/workers/training_worker.py
        TagSpecifications:
          - ResourceType: instance
            Tags:
              - Key: Name
                Value: !Sub '${ProjectName}-training-worker'
                
  TrainingWorkerASG:
    Type: AWS::AutoScaling::AutoScalingGroup
    Properties:
      AutoScalingGroupName: !Sub '${ProjectName}-training-workers'
      LaunchTemplate:
        LaunchTemplateId: !Ref TrainingWorkerLaunchTemplate
        Version: !GetAtt TrainingWorkerLaunchTemplate.LatestVersionNumber
      MinSize: 0
      MaxSize: 4
      DesiredCapacity: 2
      VPCZoneIdentifier:
        - !Ref PrivateSubnet1
        - !Ref PrivateSubnet2
      Tags:
        - Key: Name
          Value: !Sub '${ProjectName}-training-worker'
          PropagateAtLaunch: true
          
  # SQS Queues
  TrainingQueue:
    Type: AWS::SQS::Queue
    Properties:
      QueueName: !Sub '${ProjectName}-training-queue'
      VisibilityTimeout: 7200  # 2 hours for training
      MessageRetentionPeriod: 1209600  # 14 days
      
  EvaluationQueue:
    Type: AWS::SQS::Queue
    Properties:
      QueueName: !Sub '${ProjectName}-evaluation-queue'
      VisibilityTimeout: 600  # 10 minutes
      
  # DynamoDB Tables
  ExperimentsTable:
    Type: AWS::DynamoDB::Table
    Properties:
      TableName: !Sub '${ProjectName}-experiments'
      BillingMode: PAY_PER_REQUEST
      AttributeDefinitions:
        - AttributeName: experiment_id
          AttributeType: S
        - AttributeName: timestamp
          AttributeType: N
      KeySchema:
        - AttributeName: experiment_id
          KeyType: HASH
        - AttributeName: timestamp
          KeyType: RANGE
      GlobalSecondaryIndexes:
        - IndexName: timestamp-index
          KeySchema:
            - AttributeName: timestamp
              KeyType: HASH
          Projection:
            ProjectionType: ALL
            
  MetricsTable:
    Type: AWS::DynamoDB::Table
    Properties:
      TableName: !Sub '${ProjectName}-metrics'
      BillingMode: PAY_PER_REQUEST
      AttributeDefinitions:
        - AttributeName: metric_id
          AttributeType: S
        - AttributeName: timestamp
          AttributeType: N
      KeySchema:
        - AttributeName: metric_id
          KeyType: HASH
        - AttributeName: timestamp
          KeyType: RANGE

Outputs:
  OrchestratorPublicIP:
    Description: Orchestrator public IP
    Value: !GetAtt OrchestratorInstance.PublicIp
    
  TrainingQueueUrl:
    Description: Training queue URL
    Value: !Ref TrainingQueue
    Export:
      Name: !Sub '${ProjectName}-training-queue-url'
```

---

## 4. Deployment Scripts

**File: `scripts/deploy_framework.sh`**

```bash
#!/bin/bash
set -e

echo "==================================="
echo "AI Video Codec Framework Deployment"
echo "==================================="

# Configuration
PROJECT_NAME="ai-video-codec"
AWS_REGION="us-east-1"
STACK_NAME="${PROJECT_NAME}-stack"

# Step 1: Create S3 bucket for artifacts
echo "Step 1: Creating S3 bucket..."
aws s3 mb s3://${PROJECT_NAME}-artifacts-${AWS_ACCOUNT_ID} --region ${AWS_REGION} || true

# Step 2: Package and upload code
echo "Step 2: Packaging code..."
zip -r deployment.zip . -x "*.git*" "*.pyc" "__pycache__/*"
aws s3 cp deployment.zip s3://${PROJECT_NAME}-artifacts-${AWS_ACCOUNT_ID}/

# Step 3: Deploy CloudFormation stack
echo "Step 3: Deploying infrastructure..."
aws cloudformation deploy \
    --template-file infrastructure/cloudformation/compute.yaml \
    --stack-name ${STACK_NAME} \
    --parameter-overrides \
        ProjectName=${PROJECT_NAME} \
        Environment=production \
    --capabilities CAPABILITY_IAM \
    --region ${AWS_REGION}

# Step 4: Upload video assets
echo "Step 4: Uploading test videos..."
aws s3 cp data/source_4k60_10s.mp4 s3://${PROJECT_NAME}-videos/source.mp4
aws s3 cp data/hevc_reference.mp4 s3://${PROJECT_NAME}-videos/reference.mp4

# Step 5: Initialize DynamoDB tables
echo "Step 5: Initializing database..."
python scripts/init_database.py

# Step 6: Start orchestrator
echo "Step 6: Starting orchestrator..."
ORCHESTRATOR_IP=$(aws cloudformation describe-stacks \
    --stack-name ${STACK_NAME} \
    --query 'Stacks[0].Outputs[?OutputKey==`OrchestratorPublicIP`].OutputValue' \
    --output text)

ssh ec2-user@${ORCHESTRATOR_IP} << 'EOF'
    cd /opt/ai-video-codec
    nohup python3 orchestrator/master.py > orchestrator.log 2>&1 &
    echo "Orchestrator started with PID: $!"
EOF

echo "==================================="
echo "Deployment complete!"
echo "Orchestrator IP: ${ORCHESTRATOR_IP}"
echo "Monitor progress: ssh ec2-user@${ORCHESTRATOR_IP}"
echo "==================================="
```

---

## 5. Monitoring & Alerting

**File: `orchestrator/reporter.py`**

```python
class HourlyReporter:
    """
    Generates comprehensive hourly progress reports.
    """
    
    def __init__(self):
        self.dynamodb = boto3.resource('dynamodb')
        self.experiments_table = self.dynamodb.Table('ai-video-codec-experiments')
        self.metrics_table = self.dynamodb.Table('ai-video-codec-metrics')
        self.sns = boto3.client('sns')
        
    def generate_report(self):
        """Generate and send hourly report"""
        
        # Gather data
        current_status = self.get_current_status()
        best_results = self.get_best_results()
        cost_data = self.get_cost_data()
        experiment_stats = self.get_experiment_stats()
        
        # Format report
        report = self.format_report(
            current_status,
            best_results,
            cost_data,
            experiment_stats
        )
        
        # Save to S3
        self.save_report(report)
        
        # Send notifications
        self.send_notification(report)
        
        # Log to CloudWatch
        self.log_metrics(best_results, cost_data)
        
    def format_report(self, current, best, cost, stats):
        """Format report as text"""
        
        elapsed_hours = (datetime.now() - self.start_time).total_seconds() / 3600
        
        report = f"""
{'═'*63}
AI VIDEO CODEC - HOURLY PROGRESS REPORT
Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')} | Elapsed: {elapsed_hours:.1f} hours
{'═'*63}

CURRENT STATUS: {current['status']}
  ├─ Architecture: {current['architecture']}
  ├─ Progress: {current['progress']}%
  └─ ETA: {current['eta']}

BEST RESULTS SO FAR:
  ├─ Bitrate Reduction: {best['bitrate_reduction']:.1f}% vs HEVC {self._check_mark(best['bitrate_reduction'] >= 90)}
  ├─ PSNR: {best['psnr']:.1f} dB {self._check_mark(best['psnr'] >= 95)}
  ├─ Inference Speed: {best['fps']:.0f} fps @ 4K {self._check_mark(best['fps'] >= 60)}
  └─ Model Size: {best['model_size']:.0f} MB {self._check_mark(best['model_size'] <= 100)}

EXPERIMENTS COMPLETED: {stats['total']}
  ├─ Success: {stats['success']}
  ├─ Partial: {stats['partial']}
  └─ Failed: {stats['failed']}

COST TRACKING:
  ├─ This Hour: ${cost['hour']:.2f}
  ├─ Last 24h: ${cost['day']:.2f}
  ├─ Month-to-Date: ${cost['mtd']:.2f}
  ├─ Projected Monthly: ${cost['projected']:.2f} {self._check_mark(cost['projected'] < 5000)}
  └─ Budget Remaining: ${5000 - cost['mtd']:.2f} ({(1 - cost['mtd']/5000)*100:.0f}%)

NEXT ACTIONS:
"""
        for i, action in enumerate(current['next_actions'], 1):
            report += f"  {i}. {action}\n"
            
        report += '═'*63 + '\n'
        
        return report
        
    def _check_mark(self, condition):
        """Return check mark or warning symbol"""
        return '✓' if condition else '⚠️'
```

---

## 6. Testing Strategy

**File: `tests/test_codec.py`**

```python
import pytest
import torch
from codec.models.hyperprior import ScaleHyperpriorCodec

def test_codec_forward_pass():
    """Test basic forward pass"""
    model = ScaleHyperpriorCodec(channels=128)
    x = torch.randn(1, 3, 8, 256, 256)  # Batch, RGB, Frames, H, W
    
    output = model(x)
    
    assert 'x_hat' in output
    assert output['x_hat'].shape == x.shape
    assert 'likelihoods' in output

def test_codec_compress_decompress():
    """Test compression and decompression"""
    model = ScaleHyperpriorCodec(channels=128)
    model.eval()
    
    x = torch.randn(1, 3, 8, 256, 256)
    
    # Compress
    compressed = model.compress(x)
    assert 'strings' in compressed
    assert 'shape' in compressed
    
    # Decompress
    reconstructed = model.decompress(compressed['strings'], compressed['shape'])
    
    # Check shapes match
    assert reconstructed.shape == x.shape
    
    # Check quality (should be reasonable even without training)
    psnr = compute_psnr(x, reconstructed)
    assert psnr > 20.0  # Minimum sanity check

def test_bitrate_calculation():
    """Test bitrate calculation"""
    model = ScaleHyperpriorCodec(channels=128)
    model.eval()
    
    x = torch.randn(1, 3, 60, 256, 256)  # 1 second @ 60fps, 256x256
    
    compressed = model.compress(x)
    total_bytes = sum(len(s) for s in compressed['strings'][0]) + len(compressed['strings'][1])
    bitrate_mbps = (total_bytes * 8) / 1e6  # Mbps for 1 second
    
    # Should be compressed (not larger than raw)
    raw_size = x.numel() * 4  # float32
    assert total_bytes < raw_size * 0.1  # At least 10x compression

def test_real_time_performance():
    """Test inference speed"""
    model = ScaleHyperpriorCodec(channels=128)
    model.eval()
    model = model.cuda()
    
    x = torch.randn(1, 3, 1, 1920, 1080).cuda()  # Single 1080p frame
    
    # Warmup
    for _ in range(10):
        _ = model(x)
    
    # Measure
    torch.cuda.synchronize()
    start = time.time()
    
    for _ in range(100):
        _ = model(x)
    
    torch.cuda.synchronize()
    elapsed = time.time() - start
    
    fps = 100 / elapsed
    
    print(f"Inference speed: {fps:.1f} fps")
    # Should be reasonably fast (will optimize later)
    assert fps > 10.0

@pytest.mark.integration
def test_end_to_end_pipeline():
    """Test complete encode-decode pipeline"""
    # Load test video
    video_path = 'data/test_video.mp4'
    frames = load_video(video_path)
    
    # Load model
    model = load_trained_model('models/best_model.pth')
    
    # Encode
    bitstream = model.compress(frames)
    
    # Decode
    reconstructed = model.decompress(bitstream['strings'], bitstream['shape'])
    
    # Evaluate
    psnr = compute_psnr(frames, reconstructed)
    ssim = compute_ssim(frames, reconstructed)
    
    # Check quality
    assert psnr >= 95.0
    assert ssim >= 0.98
    
    # Check compression
    original_size = get_file_size(video_path)
    compressed_size = sum(len(s) for strings in bitstream['strings'] for s in strings)
    compression_ratio = original_size / compressed_size
    
    assert compression_ratio >= 10.0  # 90% reduction
```

---

## 7. Quick Start Guide

### Development Environment Setup

```bash
# Clone repository
git clone https://github.com/your-org/ai-video-codec.git
cd ai-video-codec

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install dev dependencies
pip install -r requirements-dev.txt

# Setup pre-commit hooks
pre-commit install

# Run tests
pytest tests/

# Start local orchestrator (for testing)
python orchestrator/master.py --config config/local_config.yaml
```

### AWS Deployment

```bash
# Configure AWS credentials
aws configure

# Deploy infrastructure
bash scripts/deploy_framework.sh

# Monitor progress
python scripts/monitor.py --follow

# View reports
aws s3 sync s3://ai-video-codec-reports-${AWS_ACCOUNT_ID} ./reports/
```

---

## 8. Next Steps

After deploying the framework:

1. **Monitor First Hour**: Ensure all components are working
2. **Validate Baseline**: Confirm baseline models train successfully
3. **Track Costs**: Watch cost metrics closely
4. **Adjust Strategy**: Based on early results, tune experiment strategy
5. **Scale Resources**: Increase workers if making good progress
6. **Document Results**: Keep detailed logs of what works

The framework is designed to be autonomous, but human oversight in the first 24 hours is recommended to catch any issues early.

---

**Document Version:** 1.0  
**Last Updated:** October 15, 2025  
**Status:** Ready for Implementation


