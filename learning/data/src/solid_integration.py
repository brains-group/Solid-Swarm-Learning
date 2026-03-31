import requests
from requests.auth import HTTPBasicAuth
import os

class SolidTokenClient:
    def __init__(self, pod_url, username, token_id, token_secret):
        self.pod_url = pod_url.rstrip('/')
        self.username = username
        self.token_id = token_id
        self.token_secret = token_secret
        self.session = requests.Session()
        
        # Always default to the base CSS port (3000) to bypass proxy middleware
        self.auth_url = self.pod_url.replace(':3001', ':3000') if ':3001' in self.pod_url else self.pod_url
        
        # Support both 'user1' and 'user1@localhost' formats
        self.pod_identifier = username.split('@')[0] if '@' in username else username
        self.user_pod_url = f"{self.auth_url}/{self.pod_identifier}"

    def authenticate(self):
        """Exchanges Client Credentials for a Bearer Access Token."""
        print(f"   [Solid Auth] Requesting access token for {self.pod_identifier}...")
        token_endpoint = f"{self.auth_url}/.oidc/token"
        
        token_data = {
            "grant_type": "client_credentials",
            "scope": "webid"
        }
        
        auth = HTTPBasicAuth(self.token_id, self.token_secret)
        
        try:
            response = requests.post(
                token_endpoint,
                data=token_data,
                auth=auth,
                headers={"Content-Type": "application/x-www-form-urlencoded"}
            )
            
            if response.status_code == 200:
                access_token = response.json().get("access_token")
                self.session.headers.update({'Authorization': f'Bearer {access_token}'})
                return True
            else:
                print(f"   [Solid Auth] ✗ Auth failed: {response.status_code} - {response.text}")
                return False
        except Exception as e:
            print(f"   [Solid Auth] ✗ Connection error: {e}")
            return False

    def create_container(self, container_path):
        """Ensures a container exists in the pod."""
        if not container_path.endswith('/'):
            container_path += '/'
            
        full_url = f"{self.user_pod_url}{container_path}"
        response = self.session.head(full_url)
        
        if response.status_code in [200, 204, 403]:
            return True # Already exists
            
        print(f"   [Solid Sync] Creating container at {full_url}...")
        # Create using PUT
        res = self.session.put(
            full_url, 
            headers={
                "Link": '<http://www.w3.org/ns/ldp#BasicContainer>; rel="type"',
                "Content-Type": "text/turtle"
            }
        )
        return res.status_code in [201, 204, 200]

    def upload_file(self, local_file_path, remote_file_path, content_type="text/csv"):
        """Uploads a local file to the Solid Pod."""
        if not remote_file_path.startswith('/'):
            remote_file_path = '/' + remote_file_path
            
        full_remote_url = f"{self.user_pod_url}{remote_file_path}"
        
        try:
            with open(local_file_path, 'rb') as f:
                file_data = f.read()
                
            response = self.session.put(
                full_remote_url,
                data=file_data,
                headers={"Content-Type": content_type}
            )
            
            if response.status_code in [200, 201, 204, 205]:
                #print(f"   [Solid Sync] ✓ Uploaded {os.path.basename(local_file_path)} to Pod.")
                return True
            else:
                #print(f"   [Solid Sync] ✗ Failed to upload {os.path.basename(local_file_path)}. Status: {response.status_code}")
                return False
        except Exception as e:
            print(f"   [Solid Sync] ✗ Upload error: {e}")
            return False

    def download_file(self, remote_file_path, local_save_path):
        """Downloads a protected file from the Solid Pod."""
        if not remote_file_path.startswith('/'):
            remote_file_path = '/' + remote_file_path
            
        full_remote_url = f"{self.user_pod_url}{remote_file_path}"
        
        try:
            response = self.session.get(full_remote_url)
            if response.status_code == 200:
                with open(local_save_path, 'wb') as f:
                    f.write(response.content)
                #print(f"   [Solid Sync] ✓ Downloaded {os.path.basename(local_save_path)} from Pod.")
                return True
            else:
                print(f"   [Solid Sync] ✗ Failed to download. Status: {response.status_code}")
                return False
        except Exception as e:
            print(f"   [Solid Sync] ✗ Download error: {e}")
            return False