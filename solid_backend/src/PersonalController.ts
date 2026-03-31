/**
 * This file uses an Express Router to define the API endpoints. It handles
 * authentication and performs asynchronous CRUD (Create, Read, Update, Delete)
 * operations on a Solid Pod using the modern async/await syntax.
 */
import { Router, Request, Response } from 'express';
import { Session } from '@inrupt/solid-client-authn-node';
import {
  getSolidDataset,
  saveSolidDatasetAt,
  deleteSolidDataset,
  getPodUrlAll,
  saveFileInContainer,
} from '@inrupt/solid-client';
import { Personal } from './Personal.js';
import multer from 'multer';

export const personalRouter = Router();
const upload = multer({ storage: multer.memoryStorage() }); // Middleware for handling file uploads.

// --- Authentication ---
// In a real application, this would involve a full OIDC login flow. For this
// server-side script, we use client credentials as in the Java example.
async function getAuthenticatedSession(): Promise<Session> {
  const session = new Session();
  await session.login({
    oidcIssuer: process.env.MY_SOLID_IDP!,
    clientId: process.env.MY_SOLID_CLIENT_ID!,
    clientSecret: process.env.MY_SOLID_CLIENT_SECRET!,
    tokenType: "DPoP" // DPoP is more secure and generally preferred.
  });
  if (!session.info.isLoggedIn) {
      throw new Error("Authentication failed. Check your environment variables.");
  }
  return session;
}

// GET /api/pods?webid=...
personalRouter.get('/pods', async (req: Request, res: Response) => {
  try {
    const webID = req.query.webid as string;
    const session = await getAuthenticatedSession();
    // Get all Pod URLs associated with a given WebID.
    const podUrls = await getPodUrlAll(webID, { fetch: session.fetch });
    res.status(200).json(podUrls);
  } catch (error: any) {
    res.status(500).json({ message: "Failed to fetch pods", error: error.message });
  }
});

// POST /api/persons/create
personalRouter.post('/persons/create', async (req: Request, res: Response) => {
    try {
        const { identifier, ...data } = req.body;
        if (!identifier || !data.name) {
          return res.status(400).json({ message: "Missing 'identifier' or 'name' in request body." });
        }
        const session = await getAuthenticatedSession();
        
        const newPerson = Personal.build(identifier, data);

        // Save the new RDF data to the specified URL on the Pod.
        const savedDataset = await saveSolidDatasetAt(identifier, newPerson.dataset, { fetch: session.fetch });
        
        res.status(201).json({ message: "Person created successfully.", location: savedDataset.internal_resourceInfo.sourceIri });
    } catch (error: any) {
        res.status(500).json({ message: "Failed to create person", error: error.message });
    }
});
// ... Other endpoints (GET, DELETE, PUT for non-RDF files) would be similarly converted.