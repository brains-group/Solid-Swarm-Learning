/**
 * Automates the creation of Solid accounts, WebIDs, Pods, and Client Credentials
 * on a local Solid Community Server instance, exporting them to user_accounts.json.
 */
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// --- NEW: Python Config Parser ---
function getNumClients() {
  try {
    // Navigate from solid_backend/scripts back to Swarm root, then into learning/data
    const configPath = path.join(__dirname, '../../learning/data/src/config.py');
    const configContent = fs.readFileSync(configPath, 'utf8');
    
    // Look for the exact line: NUM_ACTIVE_CLIENTS = X
    const match = configContent.match(/NUM_ACTIVE_CLIENTS\s*=\s*(\d+)/);
    
    if (match && match[1]) {
      return parseInt(match[1], 10);
    } else {
      console.warn('⚠️ Could not find NUM_ACTIVE_CLIENTS in config.py. Defaulting to 10.');
    }
  } catch (err) {
    console.warn(`⚠️ Could not read config.py (${err.message}). Defaulting to 10.`);
  }
  return 10; // Failsafe default
}

async function createSolidAccountAndPod(baseUrl, email, password, podName) {
  const idpIndex = `${baseUrl}/.account/`;

  const fetchJson = async (url, options = {}) => {
    const headers = { ...options.headers, 'Accept': 'application/json' };
    const res = await fetch(url, { ...options, headers });
    
    if (!res.ok) throw new Error(`HTTP ${res.status}: ${await res.text()}`);
    
    const contentType = res.headers.get('content-type') || '';
    if (contentType.includes('text/html')) {
      const text = await res.text();
      throw new Error(`Expected JSON but got HTML from . Proxy might be intercepting.\nPreview: ${text.substring(0, 100)}`);
    }
    return res.json();
  };

  // 1. Initial GET to retrieve the 'create session' endpoint URL
  const data1 = await fetchJson(idpIndex);
  const createSessionUrl = data1.controls.account.create;

  // 2. POST to start an account registration session
  const res2 = await fetch(createSessionUrl, { method: 'POST' });
  if (!res2.ok) throw new Error(`Failed to create account session: ${res2.statusText}`);
  
  let cookieString = '';
  if (res2.headers.getSetCookie) {
    cookieString = res2.headers.getSetCookie().map(c => c.split(';')[0]).join('; ');
  } else {
    const setCookie = res2.headers.get('set-cookie');
    if (setCookie) cookieString = setCookie.split(';')[0];
  }

  // 3. GET controls again using the session cookie
  const data3 = await fetchJson(idpIndex, { headers: { cookie: cookieString } });

  // 4. POST to register the credentials (Email & Password)
  await fetchJson(data3.controls.password.create, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', cookie: cookieString },
    body: JSON.stringify({ email, password, confirmPassword: password })
  });

  // 5. POST to create the Pod and automatic WebID
  const result = await fetchJson(data3.controls.account.pod, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', cookie: cookieString },
    body: JSON.stringify({ name: podName })
  });

  // 6. GET controls a final time to unlock the Token Generation endpoint, and POST to generate a token
  const data6 = await fetchJson(idpIndex, { headers: { cookie: cookieString } });
  const tokenResult = await fetchJson(data6.controls.account.clientCredentials, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', cookie: cookieString },
    body: JSON.stringify({ name: `-automated-token`, webId: result.webId })
  });

  return { podResult: result, tokenResult };
}

async function generateAccounts() {
  const BASE_URL = 'http://localhost:3000';
  const accountsDict = {};

  for (let i = 1; i <= getNumClients(); i++) { 
    const username = `user${i}`;
    // THE FIX: Use @example.com so the Solid Server's email validation passes
    const email = `user${i}@example.com`; 
    const password = `pass${i}`;

    try {
      const { podResult, tokenResult } = await createSolidAccountAndPod(BASE_URL, email, password, username);
      console.log(`✅ Created Account ${username}: [Email: ${email}]`);
      
      accountsDict[username] = { 
          email, 
          password, 
          pod: podResult.pod, 
          webid: podResult.webId, 
          client_credentials_token_identifier: tokenResult.id, 
          client_credentials_token_secret: tokenResult.secret 
      };
    } catch (error) { 
        console.error(`❌ Failed to create account ${username}:`, error.message); 
    }
  }

  const outputPath = path.join(__dirname, '../../secret/user_accounts.json');
  
  // Ensure the target directory exists before writing
  const outputDir = path.dirname(outputPath);
  if (!fs.existsSync(outputDir)){
      fs.mkdirSync(outputDir, { recursive: true });
  }

  fs.writeFileSync(outputPath, JSON.stringify(accountsDict, null, 2));
  console.log(`\n💾 Successfully wrote all credentials to: ${outputPath}`);
}

generateAccounts();