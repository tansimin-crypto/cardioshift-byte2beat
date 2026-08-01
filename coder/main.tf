terraform {
  required_version = ">= 1.5.0"

  required_providers {
    coder = {
      source  = "coder/coder"
      version = ">= 2.0.0, < 3.0.0"
    }
    docker = {
      source  = "kreuzwerker/docker"
      version = ">= 3.0.0, < 4.0.0"
    }
  }
}

variable "repo_url" {
  description = "Private SSH Git repository cloned with the Coder user key."
  type        = string
  default     = "git@github.com:tansimin-crypto/byte-to-beat.git"
}

data "coder_external_auth" "github" {
  id = "github"
}

data "coder_parameter" "repo_ref" {
  name         = "repo_ref"
  display_name = "Repository ref"
  description  = "Branch, tag, or commit fetched from origin."
  type         = "string"
  default      = "codex/submission-closure"
  mutable      = true
  order        = 1
}

data "coder_parameter" "expected_release_sha" {
  name         = "expected_release_sha"
  display_name = "Expected release SHA"
  description  = "Exact 40-character Git commit that must run."
  type         = "string"
  default      = "0000000000000000000000000000000000000000"
  mutable      = true
  order        = 2
}

data "coder_workspace" "me" {}
data "coder_workspace_owner" "me" {}

resource "coder_agent" "main" {
  arch = "amd64"
  os   = "linux"

  env = {
    CARDIOSHIFT_REPO_URL             = var.repo_url
    CARDIOSHIFT_REPO_REF             = data.coder_parameter.repo_ref.value
    CARDIOSHIFT_EXPECTED_RELEASE_SHA = data.coder_parameter.expected_release_sha.value
  }

  startup_script = file("${path.module}/startup.sh")

  metadata {
    display_name = "Core tests"
    key          = "cardioshift_tests"
    script       = "test -f /workspace/.coder-status/tests.ok && echo passed || echo pending"
    interval     = 10
    timeout      = 2
  }

  metadata {
    display_name = "Release SHA"
    key          = "cardioshift_release_sha"
    script       = "git -C /workspace/byte-to-beat rev-parse HEAD 2>/dev/null || echo pending"
    interval     = 10
    timeout      = 2
  }
}

resource "coder_app" "jupyter" {
  agent_id     = coder_agent.main.id
  slug         = "jupyter"
  display_name = "JupyterLab"
  icon         = "/icon/jupyter.svg"
  url          = "http://localhost:8888"
  subdomain    = false
  share        = "owner"

  healthcheck {
    url       = "http://localhost:8888/api"
    interval  = 5
    threshold = 12
  }
}

resource "coder_app" "streamlit" {
  agent_id     = coder_agent.main.id
  slug         = "cardioshift"
  display_name = "CardioShift"
  icon         = "/icon/streamlit.svg"
  url          = "http://localhost:8501"
  subdomain    = false
  share        = "owner"

  healthcheck {
    url       = "http://localhost:8501/_stcore/health"
    interval  = 5
    threshold = 12
  }
}

resource "docker_image" "workspace" {
  name = "cardioshift-coder:${data.coder_workspace.me.id}"

  build {
    context    = "${path.module}/.."
    dockerfile = "coder/Dockerfile"
  }
}

resource "docker_volume" "workspace" {
  name = "cardioshift-${data.coder_workspace.me.id}"

  lifecycle {
    ignore_changes = all
  }
}

locals {
  agent_init_script = replace(
    replace(coder_agent.main.init_script, "127.0.0.1", "host.docker.internal"),
    "localhost",
    "host.docker.internal",
  )
}

resource "docker_container" "workspace" {
  count = data.coder_workspace.me.start_count
  image = docker_image.workspace.image_id
  name  = "cardioshift-${data.coder_workspace_owner.me.name}-${lower(data.coder_workspace.me.name)}"

  command = ["sh", "-c", local.agent_init_script]

  env = [
    "CODER_AGENT_TOKEN=${coder_agent.main.token}",
    "CARDIOSHIFT_REPO_URL=${var.repo_url}",
    "CARDIOSHIFT_REPO_REF=${data.coder_parameter.repo_ref.value}",
    "CARDIOSHIFT_EXPECTED_RELEASE_SHA=${data.coder_parameter.expected_release_sha.value}",
  ]

  host {
    host = "host.docker.internal"
    ip   = "host-gateway"
  }

  volumes {
    container_path = "/workspace"
    volume_name    = docker_volume.workspace.name
  }

  lifecycle {
    precondition {
      condition     = can(regex("^[0-9a-f]{40}$", data.coder_parameter.expected_release_sha.value))
      error_message = "expected_release_sha must be a full lowercase 40-character Git SHA."
    }
  }
}
