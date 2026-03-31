import { Router } from 'express';
import { personalRouter } from './PersonalController.js';
import { verificationTokensRouter } from './VerificationTokensController.js';
import { tokenExchangeRouter } from './TokenExchangeController.js';
// Import any other routers you create here

const mainRouter = Router();

mainRouter.use(personalRouter);
mainRouter.use(verificationTokensRouter);
mainRouter.use(tokenExchangeRouter);
// Use other routers here

export { mainRouter };