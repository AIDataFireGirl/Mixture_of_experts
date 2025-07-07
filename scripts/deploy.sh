#!/bin/bash

# Deployment script for MoE Recommendation System
# This script handles the complete deployment process with validation and rollback capabilities

set -euo pipefail

# Configuration
NAMESPACE="moe-system"
DEPLOYMENT_NAME="moe-api"
REDIS_NAME="redis-master"
IMAGE_NAME="moe-recommendation"
IMAGE_TAG="latest"
KUBECONFIG="${KUBECONFIG:-~/.kube/config}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check if kubectl is installed
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl is not installed"
        exit 1
    fi
    
    # Check if docker is installed
    if ! command -v docker &> /dev/null; then
        log_error "docker is not installed"
        exit 1
    fi
    
    # Check if helm is installed (optional)
    if ! command -v helm &> /dev/null; then
        log_warning "helm is not installed - some features may not work"
    fi
    
    # Check kubectl connectivity
    if ! kubectl cluster-info &> /dev/null; then
        log_error "Cannot connect to Kubernetes cluster"
        exit 1
    fi
    
    log_success "Prerequisites check passed"
}

# Build Docker image
build_image() {
    log_info "Building Docker image..."
    
    # Build the image
    docker build -t "${IMAGE_NAME}:${IMAGE_TAG}" .
    
    if [ $? -eq 0 ]; then
        log_success "Docker image built successfully"
    else
        log_error "Failed to build Docker image"
        exit 1
    fi
}

# Push Docker image (if registry is specified)
push_image() {
    if [ -n "${DOCKER_REGISTRY:-}" ]; then
        log_info "Pushing Docker image to registry..."
        
        # Tag for registry
        docker tag "${IMAGE_NAME}:${IMAGE_TAG}" "${DOCKER_REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}"
        
        # Push to registry
        docker push "${DOCKER_REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}"
        
        if [ $? -eq 0 ]; then
            log_success "Docker image pushed successfully"
        else
            log_error "Failed to push Docker image"
            exit 1
        fi
    else
        log_warning "No Docker registry specified - skipping push"
    fi
}

# Create namespace
create_namespace() {
    log_info "Creating namespace ${NAMESPACE}..."
    
    kubectl create namespace "${NAMESPACE}" --dry-run=client -o yaml | kubectl apply -f -
    
    log_success "Namespace created/updated"
}

# Deploy Redis
deploy_redis() {
    log_info "Deploying Redis..."
    
    # Apply Redis StatefulSet
    kubectl apply -f deployment/k8s-deployment.yaml -n "${NAMESPACE}"
    
    # Wait for Redis to be ready
    log_info "Waiting for Redis to be ready..."
    kubectl wait --for=condition=ready pod -l app=redis -n "${NAMESPACE}" --timeout=300s
    
    log_success "Redis deployed successfully"
}

# Deploy MoE API
deploy_moe_api() {
    log_info "Deploying MoE API..."
    
    # Apply MoE API deployment
    kubectl apply -f deployment/k8s-deployment.yaml -n "${NAMESPACE}"
    
    # Wait for deployment to be ready
    log_info "Waiting for MoE API deployment to be ready..."
    kubectl wait --for=condition=available deployment/${DEPLOYMENT_NAME} -n "${NAMESPACE}" --timeout=600s
    
    log_success "MoE API deployed successfully"
}

# Deploy monitoring
deploy_monitoring() {
    log_info "Deploying monitoring stack..."
    
    # Create monitoring namespace
    kubectl create namespace monitoring --dry-run=client -o yaml | kubectl apply -f -
    
    # Deploy Prometheus (if helm is available)
    if command -v helm &> /dev/null; then
        # Add Prometheus Helm repository
        helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
        helm repo update
        
        # Install Prometheus
        helm install prometheus prometheus-community/kube-prometheus-stack \
            --namespace monitoring \
            --create-namespace \
            --set prometheus.prometheusSpec.retention=30d \
            --set prometheus.prometheusSpec.storageSpec.volumeClaimTemplate.spec.resources.requests.storage=10Gi
        
        log_success "Prometheus deployed successfully"
    else
        log_warning "Helm not available - skipping Prometheus deployment"
    fi
}

# Health check
health_check() {
    log_info "Performing health check..."
    
    # Check if pods are running
    local pod_status=$(kubectl get pods -n "${NAMESPACE}" -l app=moe-api -o jsonpath='{.items[*].status.phase}')
    
    if [[ "${pod_status}" == *"Running"* ]]; then
        log_success "MoE API pods are running"
    else
        log_error "MoE API pods are not running properly"
        return 1
    fi
    
    # Check if Redis is running
    local redis_status=$(kubectl get pods -n "${NAMESPACE}" -l app=redis -o jsonpath='{.items[*].status.phase}')
    
    if [[ "${redis_status}" == *"Running"* ]]; then
        log_success "Redis pods are running"
    else
        log_error "Redis pods are not running properly"
        return 1
    fi
    
    # Test API endpoint (if service is available)
    local service_ip=$(kubectl get service moe-api-service -n "${NAMESPACE}" -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || echo "")
    
    if [ -n "${service_ip}" ]; then
        log_info "Testing API endpoint..."
        
        # Wait for service to be ready
        sleep 30
        
        # Test health endpoint
        if curl -f "http://${service_ip}/health" &> /dev/null; then
            log_success "API health check passed"
        else
            log_warning "API health check failed - service may still be starting"
        fi
    else
        log_warning "Service IP not available - skipping API test"
    fi
}

# Rollback function
rollback() {
    log_warning "Rolling back deployment..."
    
    # Rollback to previous revision
    kubectl rollout undo deployment/${DEPLOYMENT_NAME} -n "${NAMESPACE}"
    
    # Wait for rollback to complete
    kubectl rollout status deployment/${DEPLOYMENT_NAME} -n "${NAMESPACE}"
    
    log_success "Rollback completed"
}

# Cleanup function
cleanup() {
    log_info "Cleaning up resources..."
    
    # Delete namespace (this will delete all resources in the namespace)
    kubectl delete namespace "${NAMESPACE}" --ignore-not-found=true
    
    # Delete monitoring namespace
    kubectl delete namespace monitoring --ignore-not-found=true
    
    log_success "Cleanup completed"
}

# Show deployment status
show_status() {
    log_info "Deployment status:"
    
    echo
    echo "=== Pods ==="
    kubectl get pods -n "${NAMESPACE}"
    
    echo
    echo "=== Services ==="
    kubectl get services -n "${NAMESPACE}"
    
    echo
    echo "=== Deployments ==="
    kubectl get deployments -n "${NAMESPACE}"
    
    echo
    echo "=== Ingress ==="
    kubectl get ingress -n "${NAMESPACE}"
    
    echo
    echo "=== Events ==="
    kubectl get events -n "${NAMESPACE}" --sort-by='.lastTimestamp'
}

# Main deployment function
deploy() {
    log_info "Starting MoE Recommendation System deployment..."
    
    # Check prerequisites
    check_prerequisites
    
    # Build and push image
    build_image
    push_image
    
    # Deploy infrastructure
    create_namespace
    deploy_redis
    deploy_moe_api
    
    # Deploy monitoring (optional)
    if [ "${DEPLOY_MONITORING:-true}" = "true" ]; then
        deploy_monitoring
    fi
    
    # Health check
    health_check
    
    # Show status
    show_status
    
    log_success "Deployment completed successfully!"
    
    echo
    echo "=== Access Information ==="
    echo "API Endpoint: http://$(kubectl get service moe-api-service -n "${NAMESPACE}" -o jsonpath='{.status.loadBalancer.ingress[0].ip}')"
    echo "Health Check: http://$(kubectl get service moe-api-service -n "${NAMESPACE}" -o jsonpath='{.status.loadBalancer.ingress[0].ip}')/health"
    echo "Metrics: http://$(kubectl get service moe-api-service -n "${NAMESPACE}" -o jsonpath='{.status.loadBalancer.ingress[0].ip}'):8001/metrics"
}

# Usage function
usage() {
    echo "Usage: $0 [COMMAND]"
    echo
    echo "Commands:"
    echo "  deploy     - Deploy the complete MoE system"
    echo "  build      - Build Docker image only"
    echo "  status     - Show deployment status"
    echo "  health     - Perform health check"
    echo "  rollback   - Rollback to previous deployment"
    echo "  cleanup    - Clean up all resources"
    echo "  help       - Show this help message"
    echo
    echo "Environment variables:"
    echo "  DOCKER_REGISTRY     - Docker registry for pushing images"
    echo "  DEPLOY_MONITORING   - Deploy monitoring stack (default: true)"
    echo "  KUBECONFIG          - Path to kubeconfig file"
}

# Main script logic
case "${1:-deploy}" in
    deploy)
        deploy
        ;;
    build)
        check_prerequisites
        build_image
        ;;
    status)
        show_status
        ;;
    health)
        health_check
        ;;
    rollback)
        rollback
        ;;
    cleanup)
        cleanup
        ;;
    help|--help|-h)
        usage
        ;;
    *)
        log_error "Unknown command: $1"
        usage
        exit 1
        ;;
esac 