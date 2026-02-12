# Elasticsearch Setup for StateLM

StateLM requires an Elasticsearch instance for the `searchEngine` tool. Below are two methods to set it up: using **Docker** (recommended) or a **manual installation**.

---

## Method 1: Docker Setup

### 1. Create a Docker network and pull the image

```bash
docker network create elastic
docker pull docker.elastic.co/elasticsearch/elasticsearch:9.1.5
```

### 2. Configure virtual memory

Elasticsearch requires `vm.max_map_count` to be at least `262144`.

```bash
# Check the current value
cat /proc/sys/vm/max_map_count

# Set to the required minimum (effective until next reboot)
sysctl -w vm.max_map_count=262144
```

### 3. Run Elasticsearch

```bash
docker run --name es01 --net elastic \
  -p 9200:9200 \
  -it -m 6GB \
  -e "xpack.ml.use_auto_machine_memory_percent=true" \
  docker.elastic.co/elasticsearch/elasticsearch:9.1.5
```

### 4. Retrieve credentials and TLS certificate

On first startup, Elasticsearch prints a generated password. Export it and copy the CA certificate from the container:

```bash
export ES_PASS="<password_from_startup_output>"
docker cp es01:/usr/share/elasticsearch/config/certs/http_ca.crt .
```

### 5. Verify the connection

```bash
curl --cacert ./http_ca.crt -u elastic:$ES_PASS https://localhost:9200
```

---

## Method 2: Manual Installation

### 1. Download and extract Elasticsearch

Download the Elasticsearch 9.1.5 tarball and extract it:

```bash
cd /home
wget https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-9.1.5-linux-x86_64.tar.gz
tar -xzf elasticsearch-9.1.5-linux-x86_64.tar.gz
cd elasticsearch-9.1.5/
```

### 2. Create a dedicated user

Elasticsearch should not run as root. Create an `elastic` user and assign ownership:

```bash
sudo useradd -m -s /bin/bash elastic || true   # ignore if user already exists
sudo chown -R elastic:elastic /home/elasticsearch-9.1.5
```

### 3. Start Elasticsearch

On first startup, Elasticsearch will generate a password for the `elastic` user and enable TLS automatically. Save the output.

```bash
sudo -u elastic /home/elasticsearch-9.1.5/bin/elasticsearch &
```

### 4. Retrieve credentials and TLS certificate

Note the password printed during first startup, then export it. The auto-generated CA certificate is located in the `config/certs` directory:

```bash
export ES_PASS="<password_from_startup_output>"
cp /home/elasticsearch-9.1.5/config/certs/http_ca.crt .
```

### 5. Verify the connection

Wait a few seconds for the service to start, then test:

```bash
curl --cacert ./http_ca.crt -u elastic:$ES_PASS https://localhost:9200
```

---

## Set environment variables

Export the following so that StateLM can connect to Elasticsearch:

```bash
export ES_HOST="https://localhost:9200"
export ES_USER="elastic"
export ES_PASS="your_password"
export ES_CA_CERT="/path/to/http_ca.crt"
```
