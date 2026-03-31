/**
 * Conversion of: VerificationTokens.java
 * This class models the Verification Token RDF data. A key detail from the
 * original Java code is that it always modifies the Thing with the subject URI
 * "https://id.inrupt.com/cartal", regardless of the resource's own identifier.
 * This model preserves that logic.
 */
import {
  createThing,
  getThing,
  setThing,
  getStringNoLocale,
  setStringNoLocale,
  Thing,
  SolidDataset,
  WithServerResourceInfo,
  createSolidDataset,
} from '@inrupt/solid-client';

// The specific predicate for the OIDC registration token.
const SOLID_TERMS = {
  oidcIssuerRegistrationToken: "http://www.w3.org/ns/solid/terms#oidcIssuerRegistrationToken",
};

// The hardcoded subject URL that will be modified within the dataset.
const SUBJECT_URL = "https://id.inrupt.com/cartal";

export class VerificationTokens {
  public dataset: SolidDataset;
  private thing: Thing;
  public readonly resourceIdentifier: string;

  constructor(resourceIdentifier: string, dataset?: SolidDataset & WithServerResourceInfo) {
    this.resourceIdentifier = resourceIdentifier;
    this.dataset = dataset ?? createSolidDataset();

    // IMPORTANT: We get or create the Thing for the *hardcoded subject URL*.
    const existingThing = getThing(this.dataset, SUBJECT_URL);
    this.thing = existingThing ?? createThing({ url: SUBJECT_URL });
    this.dataset = setThing(this.dataset, this.thing);
  }

  private updateThing(newThing: Thing): void {
    this.thing = newThing;
    this.dataset = setThing(this.dataset, this.thing);
  }

  /**
   * Reads the token value (the pivot) from the Thing.
   */
  getPivot(): string | null {
    return getStringNoLocale(this.thing, SOLID_TERMS.oidcIssuerRegistrationToken);
  }

  /**
   * Sets the token value (the pivot) on the Thing.
   */
  setPivot(pivotValue: string): void {
    const newThing = setStringNoLocale(this.thing, SOLID_TERMS.oidcIssuerRegistrationToken, pivotValue);
    this.updateThing(newThing);
  }
}