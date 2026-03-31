/**
 * This class models the Personal RDF data. Instead of extending a base class,
 * it encapsulates a SolidDataset and a Thing. The getters and setters use
 * functions from @inrupt/solid-client to interact with the RDF data in a
 * type-safe and idiomatic JavaScript way.
 */
import {
  createThing,
  getThing,
  setThing,
  getStringNoLocale,
  setStringNoLocale,
  getUrl,
  setUrl,
  getDatetime,
  setDatetime,
  addUrl,
  buildThing,
  Thing,
  SolidDataset,
  //WithServerResourceInfo,
  createSolidDataset,
} from '@inrupt/solid-client';

// Define the RDF vocabulary terms (predicates).
const RDF = {
  type: "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
};

const SCHEMA_ORG = {
  birthDate: "https://schema.org/birthDate",
  name: "https://schema.org/name",
  gender: "https://schema.org/gender",
  pronouns: "https://schema.org/pronouns",
  priceCurrency: "https://schema.org/priceCurrency",
  country: "https://schema.org/country",
  image: "https://schema.org/image",
  nid: "https://schema.org/nid",
  Invoice: "https://schema.org/Invoice",
};

export class Personal {
  public dataset: SolidDataset;
  private thing: Thing;
  public readonly identifier: string;

  constructor(identifier: string, dataset?: SolidDataset) {
    this.identifier = identifier;
    this.dataset = dataset ?? createSolidDataset();

    // Find the main data "Thing" in the dataset or create it if it's new.
    const existingThing = getThing(this.dataset, identifier);
    this.thing = existingThing ?? createThing({ url: identifier });
    this.dataset = setThing(this.dataset, this.thing);
  }

  // Helper to update the internal state when a property changes.
  private updateThing(newThing: Thing): void {
    this.thing = newThing;
    this.dataset = setThing(this.dataset, this.thing);
  }

  // Getters and setters that wrap the @inrupt/solid-client functions.
  getName(): string | null {
    return getStringNoLocale(this.thing, SCHEMA_ORG.name);
  }

  setName(name: string): void {
    const newThing = setStringNoLocale(this.thing, SCHEMA_ORG.name, name);
    this.updateThing(newThing);
  }
  
  getDateOfBirth(): Date | null {
    return getDatetime(this.thing, SCHEMA_ORG.birthDate);
  }
  
  setDateOfBirth(date: Date): void {
    const newThing = setDatetime(this.thing, SCHEMA_ORG.birthDate, date);
    this.updateThing(newThing);
  }

  // ... other getters/setters would follow the same pattern
  
  addId(idUrl: string): void {
     this.updateThing(addUrl(this.thing, SCHEMA_ORG.image, idUrl));
  }
  
  /**
   * Factory method to create a new Personal object with initial data,
   * replacing the complex @JsonCreator constructor from the Java code.
   */
  public static build(identifier: string, data: { name: string; dateOfBirth: Date; /* other fields */ }) {
      const personal = new Personal(identifier);
      const thingBuilder = buildThing(personal.thing)
          .addUrl(RDF.type, SCHEMA_ORG.Invoice)
          .addStringNoLocale(SCHEMA_ORG.name, data.name)
          .addDatetime(SCHEMA_ORG.birthDate, data.dateOfBirth);
          // ... add other properties here
          
      personal.updateThing(thingBuilder.build());
      return personal;
  }
}