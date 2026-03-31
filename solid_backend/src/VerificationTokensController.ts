/**
 * Conversion of: VerificationTokensController.java
 * This file defines the API endpoints for managing VerificationTokens using an
 * Express Router. It handles authentication and performs async CRUD operations.
 */
import { Router, Request, Response } from 'express';
import { Session } from '@inrupt/solid-client-authn-node';
import {
  getSolidDataset,
  saveSolidDatasetAt,
  deleteFile, // Note: deleting a resource uses deleteFile
} from '@inrupt/solid-client';
import { VerificationTokens } from './VerificationTokens.js'; 

export const verificationTokensRouter = Router();

// --- Authentication (reused logic) ---
async function getAuthenticatedSession(): Promise<Session> {
  const session = new Session();
  await session.login({
    oidcIssuer: process.env.MY_SOLID_IDP!,
    clientId: process.env.MY_SOLID_CLIENT_ID!,
    clientSecret: process.env.MY_SOLID_CLIENT_SECRET!,
    tokenType: "DPoP"
  });
  if (!session.info.isLoggedIn) {
      throw new Error("Authentication failed. Check your environment variables.");
  }
  return session;
}

// POST /api/tokens/create
verificationTokensRouter.post('/tokens/create', async (req: Request, res: Response) => {
  try {
    const { identifier, pivot } = req.body;
    if (!identifier || !pivot) {
      return res.status(400).json({ message: "Missing 'identifier' or 'pivot' in request body." });
    }
    const session = await getAuthenticatedSession();
    
    // Create a new model instance. This prepares an empty dataset.
    const newToken = new VerificationTokens(identifier);
    // Set the data on the specific Thing inside the dataset.
    newToken.setPivot(pivot);

    // Save the entire dataset to the pod.
    const savedDataset = await saveSolidDatasetAt(identifier, newToken.dataset, { fetch: session.fetch });

    res.status(201).json({ 
        message: "Token created successfully.", 
        location: savedDataset.internal_resourceInfo.sourceIri,
        pivot: new VerificationTokens(identifier, savedDataset).getPivot()
    });
  } catch (error: any) {
    res.status(500).json({ message: "Failed to create token", error: error.message });
  }
});

// GET /api/tokens/get?resourceURL=...
verificationTokensRouter.get('/tokens/get', async (req: Request, res: Response) => {
  try {
    const resourceURL = req.query.resourceURL as string;
    if (!resourceURL) {
      return res.status(400).json({ message: "Missing 'resourceURL' query parameter." });
    }
    const session = await getAuthenticatedSession();

    const fetchedDataset = await getSolidDataset(resourceURL, { fetch: session.fetch });
    const token = new VerificationTokens(resourceURL, fetchedDataset);
    
    res.status(200).json({ pivot: token.getPivot() });
  } catch (error: any) {
    res.status(500).json({ message: "Failed to get token", error: error.message });
  }
});

// DELETE /api/tokens/delete?resourceURL=...
verificationTokensRouter.delete('/tokens/delete', async (req: Request, res: Response) => {
    try {
        const resourceURL = req.query.resourceURL as string;
        if (!resourceURL) {
            return res.status(400).json({ message: "Missing 'resourceURL' query parameter." });
        }
        const session = await getAuthenticatedSession();

        // Deleting a resource (whether it contains RDF or not) uses deleteFile.
        await deleteFile(resourceURL, { fetch: session.fetch });

        res.status(204).send(); // Success, No Content
    } catch (error: any) {
        res.status(500).json({ message: "Failed to delete token", error: error.message });
    }
});