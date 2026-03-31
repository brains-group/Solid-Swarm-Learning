/**
 * Token Exchange Controller
 * 
 * Provides an API endpoint to exchange session cookies for Bearer access tokens.
 * This allows the browser to authenticate with pod resources that require Bearer tokens.
 * 
 * Security: Client credentials are stored server-side and never exposed to the browser.
 */
import { Router, Request, Response } from 'express';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

export const tokenExchangeRouter = Router();

interface UserCredentials {
    email: string;
    password?: string;
    pod?: string;
    webid?: string;
    client_credentials_token_identifier?: string;
    client_credentials_token_secret?: string;
}

/**
 * Load user credentials from the secret file
 */
function loadUserCredentials(username: string): UserCredentials | null {
    const credentialsPath = path.join(__dirname, '../secret/user-accounts.json');
    
    try {
        if (!fs.existsSync(credentialsPath)) {
            console.error(`Credentials file not found: ${credentialsPath}`);
            return null;
        }
        
        const credentialsData = fs.readFileSync(credentialsPath, 'utf-8');
        const credentials = JSON.parse(credentialsData);
        
        return credentials[username] || null;
    } catch (error) {
        console.error(`Error loading credentials for ${username}:`, error);
        return null;
    }
}

/**
 * Exchange client credentials for an access token
 */
async function getAccessToken(
    podUrl: string,
    tokenIdentifier: string,
    tokenSecret: string,
    webid?: string
): Promise<string> {
    const podUrlClean = podUrl.replace(/\/$/, '');
    
    // Discover token endpoint from account controls
    let tokenEndpoint = `${podUrlClean}/.oidc/token`; // Default
    
    try {
        const accountResponse = await fetch(`${podUrlClean}/.account/`);
        if (accountResponse.ok) {
            const accountData = await accountResponse.json() as any;
            tokenEndpoint = accountData.controls?.clientCredentials?.tokenEndpoint ||
                          accountData.controls?.tokenEndpoint ||
                          tokenEndpoint;
        }
    } catch (error) {
        console.warn('Could not get account controls, using default token endpoint:', error);
    }
    
    // Request access token using client credentials grant
    const tokenData = new URLSearchParams({
        grant_type: 'client_credentials',
        scope: 'webid'
    });
    
    if (webid) {
        tokenData.append('webid', webid);
    }
    
    // Use Basic Auth for client authentication
    const authHeader = Buffer.from(`${tokenIdentifier}:${tokenSecret}`).toString('base64');
    
    try {
        const response = await fetch(tokenEndpoint, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/x-www-form-urlencoded',
                'Authorization': `Basic ${authHeader}`
            },
            body: tokenData.toString()
        });
        
        if (response.ok) {
            const tokenResponse = await response.json() as any;
            const accessToken = tokenResponse.access_token;
            
            if (accessToken) {
                console.log('Successfully obtained access token');
                return accessToken;
            } else {
                throw new Error('Token response missing access_token');
            }
        } else {
            const errorText = await response.text();
            throw new Error(`Token request failed: ${response.status} - ${errorText}`);
        }
    } catch (error: any) {
        console.error('Error requesting access token:', error);
        throw error;
    }
}

/**
 * POST /api/token-exchange
 * 
 * Exchange session for access token.
 * 
 * Request body:
 * {
 *   "username": "user1",
 *   "serverUrl": "http://localhost:3000"
 * }
 * 
 * Response:
 * {
 *   "accessToken": "...",
 *   "expiresIn": 3600
 * }
 */
tokenExchangeRouter.post('/token-exchange', async (req: Request, res: Response) => {
    try {
        const { username, serverUrl } = req.body;
        
        if (!username) {
            return res.status(400).json({
                error: 'Missing required field: username is required'
            });
        }
        
        // Map external server URL (3001) to internal Solid server URL (3000)
        // The Solid server runs on port 3000 internally, even though it's proxied through 3001
        let solidServerUrl = serverUrl || 'http://localhost:3001';
        if (solidServerUrl.includes(':3001')) {
            solidServerUrl = solidServerUrl.replace(':3001', ':3000');
        } else if (!solidServerUrl.includes(':')) {
            solidServerUrl = 'http://localhost:3000';
        }
        
        // Load user credentials from server-side file
        const credentials = loadUserCredentials(username);
        
        if (!credentials) {
            return res.status(404).json({
                error: `User credentials not found for: ${username}`
            });
        }
        
        const tokenIdentifier = credentials.client_credentials_token_identifier;
        const tokenSecret = credentials.client_credentials_token_secret;
        const webid = credentials.webid;
        
        if (!tokenIdentifier || !tokenSecret) {
            return res.status(400).json({
                error: `User ${username} does not have client credentials token configured`
            });
        }
        
        // Exchange client credentials for access token
        // Use internal Solid server URL (3000) for server-to-server communication
        const accessToken = await getAccessToken(solidServerUrl, tokenIdentifier, tokenSecret, webid);
        
        // Return access token to browser
        // Note: In production, you might want to set an expiration time
        res.json({
            accessToken: accessToken,
            expiresIn: 3600, // Typical token expiration (1 hour)
            tokenType: 'Bearer'
        });
        
    } catch (error: any) {
        console.error('Token exchange error:', error);
        res.status(500).json({
            error: 'Failed to exchange token',
            message: error.message
        });
    }
});

