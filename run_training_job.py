#!/usr/bin/env python3
import argparse
from datetime import datetime

import boto3
import sagemaker
from sagemaker.tensorflow import TensorFlow


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--region", type=str, default="us-east-1")
    p.add_argument("--role-arn", type=str, default=None)

    p.add_argument("--instance-type", type=str, default="ml.m5.large")
    p.add_argument("--framework-version", type=str, default="2.13")
    p.add_argument("--py-version", type=str, default="py310")

    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--conv-depth", type=int, default=3)

    p.add_argument("--base-job-name", type=str, default="fashion-cnn")
    p.add_argument("--s3-prefix", type=str, default="fashion-mnist-cnn")

    return p.parse_args()


def resolve_role(sm_session: sagemaker.Session, explicit_role: str | None) -> str:
    if explicit_role:
        return explicit_role
    try:
        # Works when running inside SageMaker environment with an attached execution role
        return sagemaker.get_execution_role()
    except Exception as e:
        raise ValueError(
            "Could not resolve execution role automatically.\n"
            "Provide --role-arn explicitly (e.g., an IAM role ARN with SageMaker permissions)."
        ) from e


def main():
    args = parse_args()

    boto_session = boto3.Session(region_name=args.region)
    sm_session = sagemaker.Session(boto_session=boto_session)

    role = resolve_role(sm_session, args.role_arn)

    default_bucket = sm_session.default_bucket()
    output_path = f"s3://{default_bucket}/{args.s3_prefix}/output"

    print(f"✅ Region: {args.region}")
    print(f"✅ Default bucket: {default_bucket}")
    print(f"✅ Output path: {output_path}")
    print(f"✅ Instance type: {args.instance_type}")

    estimator = TensorFlow(
        entry_point="train.py",
        source_dir="sagemaker_src",
        role=role,
        instance_count=1,
        instance_type=args.instance_type,
        framework_version=args.framework_version,
        py_version=args.py_version,
        output_path=output_path,
        base_job_name=args.base_job_name,
        sagemaker_session=sm_session,
        hyperparameters={
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "dropout": args.dropout,
            "conv_depth": args.conv_depth,
        },
    )

    job_name = f"{args.base_job_name}-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"
    print(f"\n🚀 Launching Training Job: {job_name}\n")

    estimator.fit(job_name=job_name, wait=True, logs=True)

    print("\n✅ Training job finished.")
    print(f"TrainingJobName: {estimator.latest_training_job.name}")
    print(f"ModelData (model.tar.gz in S3): {estimator.model_data}")


if __name__ == "__main__":
    main()
