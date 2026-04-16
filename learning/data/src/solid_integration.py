import requests
from requests.auth import HTTPBasicAuth
import os
from urllib.parse import urlparse
import time
import random
from requests.exceptions import RequestException

class SolidTokenClient:
    def __init__(self, pod_url, username, token_id, token_secret):
        self.pod_url = pod_url.rstrip('/')
        self.username = username
        self.token_id = token_id
        self.token_secret = token_secret
        self.session = requests.Session()

        # Derive the server root (scheme + netloc) so we always call the
        # identity provider at the server root (e.g. http://localhost:3000)
        parsed = urlparse(self.pod_url)
        self.auth_root = f"{parsed.scheme}://{parsed.netloc}"

        # Support both 'user1' and 'user1@localhost' formats for the user path
        self.pod_identifier = username.split('@')[0] if '@' in username else username
        # The user's pod root (where files are stored)
        self.user_pod_url = f"{self.auth_root}/{self.pod_identifier}"

    def authenticate(self):
        """Exchanges Client Credentials for a Bearer Access Token."""
        print(f"   [Solid Auth] Requesting access token for {self.pod_identifier} (client id: {self.token_id})...")
        token_endpoint = f"{self.auth_root}/.oidc/token"

        token_data = {
            "grant_type": "client_credentials",
            "scope": "webid"
        }

        # First try RFC-6749 style HTTP Basic Auth
        auth = HTTPBasicAuth(self.token_id, self.token_secret)
        headers = {"Content-Type": "application/x-www-form-urlencoded"}

        max_retries = 3
        resp = None
        for attempt in range(1, max_retries + 1):
            try:
                resp = self.session.post(token_endpoint, data=token_data, auth=auth, headers=headers, timeout=20)
            except RequestException as e:
                wait = (2 ** (attempt - 1)) + random.random() * 0.5
                print(f"   [Solid Auth] ✗ Connection error when contacting {token_endpoint} (attempt {attempt}/{max_retries}): {e}. Retrying in {wait:.1f}s")
                time.sleep(wait)
                resp = None

            if resp is None:
                continue

            if resp.status_code == 200:
                access_token = resp.json().get("access_token")
                self.session.headers.update({'Authorization': f'Bearer {access_token}'})
                return True

            # If 400/401 try fallback with client creds in body
            if resp.status_code in (400, 401):
                try:
                    fallback_data = {**token_data, 'client_id': self.token_id, 'client_secret': self.token_secret}
                    fb_resp = self.session.post(token_endpoint, data=fallback_data, headers=headers, timeout=20)
                    if fb_resp.status_code == 200:
                        access_token = fb_resp.json().get('access_token')
                        self.session.headers.update({'Authorization': f'Bearer {access_token}'})
                        return True
                    else:
                        print(f"   [Solid Auth] ✗ Auth failed (fallback): {fb_resp.status_code} - {fb_resp.text}")
                except RequestException as e:
                    print(f"   [Solid Auth] ✗ Connection error during fallback auth: {e}")

            # If not successful, log and backoff before next attempt
            wait = (2 ** (attempt - 1)) + random.random() * 0.5
            print(f"   [Solid Auth] ✗ Auth attempt {attempt} returned {resp.status_code if resp is not None else 'no resp'}. Backing off {wait:.1f}s")
            time.sleep(wait)

        # Exhausted retries
        if resp is not None:
            print(f"   [Solid Auth] ✗ Auth failed after {max_retries} attempts: {resp.status_code} - {resp.text}")
        else:
            print(f"   [Solid Auth] ✗ Auth failed after {max_retries} attempts: no response")
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