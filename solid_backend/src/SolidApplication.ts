/**
* This file is the main entry point for the Node.js application. It creates
 * an Express server, which is the direct equivalent of the Spring Boot application,
 * and registers the API routes from the controller.
 */
import 'dotenv/config'; // Loads variables from .env into process.env
import * as secp256k1 from '@noble/secp256k1';
import { Request, Response, NextFunction } from 'express';
import express from 'express';
import path from 'path';
import { fileURLToPath } from 'url';
import { createProxyMiddleware } from 'http-proxy-middleware';
import { spawn } from 'child_process';
import fs from 'fs';
import { mainRouter } from './router.js';
import crypto from 'crypto';
import { verifyEvent } from 'nostr-tools';

// Get the directory name in ES modules
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Map of Public Keys to local usernames
// This is how you "attach" your generated keys to Solid users
const AUTH_MAPPING: Record<string, string> = {
  "219eb73f02ded956ec90582809c959119068196474700de02868531acb534963": "user1",
  "951008955846e24bbf76f5c2199c4a45d82dbb9cb326400ed2eb5c41a3a72c2b": "user2",
};

//debugging
console.log("IDP Loaded:", process.env.MY_SOLID_IDP);
console.log("Client ID Loaded:", process.env.MY_SOLID_CLIENT_ID);

const app = express();
const PORT = process.env.PORT || 3001;
const SOLID_SERVER_PORT = 3000;
const SOLID_SERVER_URL = `http://localhost:${SOLID_SERVER_PORT}`;

// Apply Schnorr middleware to all pod data routes
app.use('/:username/health', schnorrAuthMiddleware);

// Set Content Security Policy to allow Solid authentication
// Note: All Solid requests are now proxied through this server, so we use localhost:3001
app.use((req, res, next) => {
  res.setHeader('Content-Security-Policy', 
    "default-src 'self'; " +
    "script-src 'self' 'unsafe-eval' 'unsafe-inline' https://login.inrupt.com https://id.inrupt.com https://storage.inrupt.com http://localhost:3001; " +
    "style-src 'self' 'unsafe-inline'; " +
    "connect-src 'self' https://login.inrupt.com https://id.inrupt.com https://storage.inrupt.com https://solidcommunity.net https://solidweb.org http://localhost:3001 http://localhost:*; " +
    "frame-src 'self' https://login.inrupt.com https://id.inrupt.com http://localhost:3001; " +
    "img-src 'self' data: https: http:; " +
    "object-src 'none'; " +
    "base-uri 'self'"
  );
  // Add CORS headers for localhost
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization');
  res.setHeader('Access-Control-Allow-Credentials', 'true');
  next();
});

// Middleware to parse incoming JSON request bodies, old one.
//app.use(express.json());

// Serve static files from the public directory for the browser-based OIDC demo
app.use(express.static(path.join(__dirname, '../public')));

// Serve Solid libraries from node_modules
app.use('/node_modules', express.static(path.join(__dirname, '../node_modules')));

//Gatekeeper Middleware to intercept requests for Schnorr
async function schnorrAuthMiddleware(req: Request, res: Response, next: NextFunction) {
  const authHeader = req.headers.authorization;

  if (!authHeader || !authHeader.startsWith('Nostr ')) {
    return next();
  }

  try {
    const token = authHeader.split(' ')[1];
    const event = JSON.parse(Buffer.from(token, 'base64').toString('utf-8'));

    const isSignatureValid = verifyEvent(event);
    if (!isSignatureValid) {
        throw new Error("Invalid NIP-98 signature or event ID.");
    }

    const { pubkey, created_at, kind, tags } = event;

    // --- CRITICAL SECURITY CHECKS ---
    // 1. Verify kind is 27235 for NIP-98
    if (kind !== 27235) throw new Error("Invalid kind");
    
    // 2. Verify timestamp is recent (e.g., within 60 seconds) to prevent replay attacks
    const now = Math.floor(Date.now() / 1000);
    if (Math.abs(now - created_at) > 60) throw new Error("Token expired"); 

    // 3. Verify URL and method match the request to prevent token replay on other endpoints
    const urlTag = tags.find(tag => tag[0] === 'u');
    const methodTag = tags.find(tag => tag[0] === 'method');
    const fullUrl = `${req.protocol}://${req.get('host')}${req.originalUrl}`;

    if (!urlTag || urlTag[1] !== fullUrl) {
      throw new Error(`URL mismatch. Token URL: ${urlTag ? urlTag[1] : 'not found'}, Request URL: ${fullUrl}`);
    }
    if (!methodTag || methodTag[1] !== req.method) {
      throw new Error(`Method mismatch. Token method: ${methodTag ? methodTag[1] : 'not found'}, Request method: ${req.method}`);
    }

    // --- END SECURITY CHECKS ---

    const username = AUTH_MAPPING[pubkey];
    if (!username) {
      return res.status(403).send("Public key not registered.");
    }

    // Attach the username and pass control to the HTTP Proxy
    req.headers['user'] = username; 
    console.log(`🛡️ NIP-98 Verified: ${pubkey.substring(0,8)}... -> ${username}`);
    
    next(); // <-- IMPORTANT: Routes to the solidProxy via HTTP
  } catch (error: any) {
    console.error("❌ NIP-98 Auth Failed:", error.message);
    return res.status(401).json({ error: "Unauthorized", details: error.message });
  }
}

// Proxy all Solid server requests to the Solid Community Server
// This includes: /.account/, /.well-known/, /user1/, /user2/, etc.
// Proxy all Solid server requests to the Solid Community Server
const solidProxy = createProxyMiddleware({
  target: SOLID_SERVER_URL,
  changeOrigin: true,
  ws: true, // Enable WebSocket proxying
  onProxyReq: (proxyReq: any, req: express.Request, res: express.Response) => {
    // Preserve original host header for Solid server
    proxyReq.setHeader('Host', `localhost:${SOLID_SERVER_PORT}`);

    // If our Schnorr middleware authenticated this request, it added a 'user' header.
    // In that case, we remove the original 'Authorization' header because the
    // backend Solid Server doesn't understand 'Nostr' tokens. We are now relying
    // on the Solid Server to trust requests from localhost (i.e., this proxy).
    if (req.headers['user']) {
      proxyReq.removeHeader('Authorization');
      console.log(`[Proxy] Schnorr-auth valid for '${req.headers['user']}'. Forwarding to Solid server without Auth header.`);
    }

    // Log proxy requests for debugging
    console.log(`[Proxy] ${req.method} ${req.path} -> ${SOLID_SERVER_URL}${req.path}`);
  },
  onError: (err: Error, req: express.Request, res: express.Response) => {
    console.error('Proxy error:', err.message);
    if (!res.headersSent) {
      res.status(502).json({ error: 'Solid server proxy error', message: err.message });
    }
  }
} as any); // Type assertion needed for v3 callback types

// --- Simplified Routing Configuration ---
// 1. API routes are handled by Express first.
app.use('/api', mainRouter);

// 2. Static files are served next. This automatically handles the root '/' by serving index.html.
app.use(express.static(path.join(__dirname, '../public')));

// 3. Any request not handled above is considered a Solid request and is proxied.
// This single catch-all replaces the multiple, redundant proxy routes.
app.use('/', solidProxy);

/**
 * Start the Solid Community Server as a child process
 */
function startSolidServer() {
  const scriptDir = path.join(__dirname, '../scripts');
  const dataDir = path.join(__dirname, '../my-solid-data');
  const logsDir = path.join(__dirname, '../logs');
  
  // Ensure directories exist
  if (!fs.existsSync(dataDir)) fs.mkdirSync(dataDir, { recursive: true });
  if (!fs.existsSync(logsDir)) fs.mkdirSync(logsDir, { recursive: true });
  
  console.log('🔧 Starting Solid Community Server...');
  
  // Start Solid server using npx
// Start Solid server using npx
  const solidServer = spawn('npx', [
    '@solid/community-server',
    '-c', '@css:config/file.json',
    '-f', dataDir
  ], {
    cwd: path.join(__dirname, '..'),
    stdio: 'pipe',
    shell: true
  });
  
  solidServer.stdout.on('data', (data) => {
    const output = data.toString();
    if (output.includes('Server running') || output.includes('listening')) {
      console.log('✅ Solid Community Server started on port', SOLID_SERVER_PORT);
    }
  });
  
  solidServer.stderr.on('data', (data) => {
    const output = data.toString();
    // Solid server writes all logs to stderr, but they're not all errors
    // Filter out info/warn messages and only show actual errors
    if (output.includes('[Components.js] info:') || 
        output.includes('[Components.js] warn:') ||
        output.includes('ExperimentalWarning') || 
        output.includes('DeprecationWarning')) {
      // These are info/warn messages, not errors - log them as info
      console.log('Solid Server:', output.trim());
    } else if (output.includes('error') || output.includes('Error') || output.includes('ERROR')) {
      // Actual errors
      console.error('Solid Server error:', output);
    } else {
      // Other stderr output - log as info
      console.log('Solid Server:', output.trim());
    }
  });
  
  solidServer.on('exit', (code) => {
    if (code !== 0 && code !== null) {
      console.error(`Solid Server exited with code ${code}`);
    }
  });
  
  // Store reference for cleanup
  (app as any).solidServerProcess = solidServer;
  
  // Wait a bit for Solid server to start
  return new Promise<void>((resolve) => {
    setTimeout(() => {
      // Check if server is responding
      fetch(`${SOLID_SERVER_URL}/.account/`)
        .then(() => {
          console.log('✅ Solid server is ready');
          resolve();
        })
        .catch(() => {
          console.log('⚠️  Solid server may still be starting...');
          resolve(); // Continue anyway
        });
    }, 2000);
  });
}

// Start Solid server, then start Express
startSolidServer().then(() => {
  app.listen(PORT, () => {
    console.log('');
    console.log('🚀 ========================================');
    console.log(`🚀 Express Server: http://localhost:${PORT}`);
    console.log(`🔧 Solid Server (proxied): http://localhost:${PORT} (→ ${SOLID_SERVER_URL})`);
    console.log(`📱 Browser App: http://localhost:${PORT}`);
    console.log(`🔧 API Endpoints: http://localhost:${PORT}/api`);
    console.log('🚀 ========================================');
    console.log('');
    console.log('💡 All requests are now served through port', PORT);
    console.log('💡 Solid server runs internally on port', SOLID_SERVER_PORT);
  });
});

// Graceful shutdown
process.on('SIGTERM', () => {
  console.log('Shutting down...');
  if ((app as any).solidServerProcess) {
    (app as any).solidServerProcess.kill();
  }
  process.exit(0);
});

process.on('SIGINT', () => {
  console.log('Shutting down...');
  if ((app as any).solidServerProcess) {
    (app as any).solidServerProcess.kill();
  }
  process.exit(0);
});
