/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
*/
import { GoogleGenAI, GenerateContentResponse, Type } from "@google/genai";
import { QueryResult } from '../types';

let ai: GoogleGenAI;

export function initialize() {
    ai = new GoogleGenAI({ apiKey: process.env.API_KEY });
}

async function delay(ms: number): Promise<void> {
    return new Promise((resolve) => setTimeout(resolve, ms));
}

export async function createRagStore(displayName: string): Promise<string> {
    if (!ai) throw new Error("Gemini AI not initialized");
    const ragStore = await ai.fileSearchStores.create({ config: { displayName } });
    if (!ragStore.name) {
        throw new Error("Failed to create RAG store: name is missing.");
    }
    return ragStore.name;
}

export async function uploadToRagStore(ragStoreName: string, file: File): Promise<void> {
    if (!ai) throw new Error("Gemini AI not initialized");
    
    let op = await ai.fileSearchStores.uploadToFileSearchStore({
        fileSearchStoreName: ragStoreName,
        file: file
    });

    while (!op.done) {
        await delay(3000);
        op = await ai.operations.get({operation: op});
    }
}

export async function fileSearch(ragStoreName: string, query: string): Promise<QueryResult> {
    if (!ai) throw new Error("Gemini AI not initialized");
    const response: GenerateContentResponse = await ai.models.generateContent({
        model: 'gemini-2.5-flash',
        contents: `Based on the provided document, answer the following question: "${query}". Provide a direct answer and pinpoint the relevant information from the document. Do not ask the user to read the manual.`,
        config: {
            tools: [
                    {
                        fileSearch: {
                            fileSearchStoreNames: [ragStoreName],
                        }
                    }
                ]
        }
    });

    const groundingChunks = response.candidates?.[0]?.groundingMetadata?.groundingChunks || [];
    return {
        text: response.text,
        groundingChunks: groundingChunks,
    };
}

export async function generateExampleQuestions(ragStoreName: string): Promise<string[]> {
    if (!ai) throw new Error("Gemini AI not initialized");
    try {
        const response = await ai.models.generateContent({
            model: 'gemini-2.5-flash',
            contents: `Based on the provided documents, generate 4 short, practical example questions a user might ask about each main product described. Return the questions as a single flat JSON array of strings. For example: ["Question 1 about Product A?", "Question 2 about Product A?", "Question 1 about Product B?"]`,
            config: {
                tools: [
                    {
                        fileSearch: {
                            fileSearchStoreNames: [ragStoreName],
                        }
                    }
                ],
                responseMimeType: 'application/json',
                responseSchema: {
                    type: Type.ARRAY,
                    items: {
                        type: Type.STRING
                    }
                }
            }
        });
        
        let jsonText = response.text.trim();
        const parsedData = JSON.parse(jsonText);
        
        if (Array.isArray(parsedData) && parsedData.every(item => typeof item === 'string')) {
            return parsedData;
        }
        
        console.warn("Received unexpected format for example questions:", parsedData);
        return [];
    } catch (error) {
        console.error("Failed to generate or parse example questions:", error);
         return [
            "What is the warranty policy?",
            "How do I clean the product?",
            "What are the main safety warnings?",
            "How do I troubleshoot common issues?"
        ];
    }
}


export async function deleteRagStore(ragStoreName: string): Promise<void> {
    if (!ai) throw new Error("Gemini AI not initialized");
    await ai.fileSearchStores.delete({
        name: ragStoreName,
        config: { force: true },
    });
}