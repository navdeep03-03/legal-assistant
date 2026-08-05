import type { AskResponse, DocumentRecord, Health } from './types'

const API_BASE = import.meta.env.VITE_API_BASE || '/api/v1'
const identityHeaders = {
  'X-Tenant-ID': 'demo-tenant',
  'X-User-ID': 'demo-user',
}

async function readResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const payload = await response.json().catch(() => ({ detail: 'Request failed' }))
    throw new Error(typeof payload.detail === 'string' ? payload.detail : 'Request failed')
  }
  if (response.status === 204) return undefined as T
  return response.json() as Promise<T>
}

export async function getHealth(): Promise<Health> {
  const response = await fetch(`${API_BASE}/health`, { headers: identityHeaders })
  return readResponse<Health>(response)
}

export async function getDocuments(): Promise<DocumentRecord[]> {
  const response = await fetch(`${API_BASE}/documents`, { headers: identityHeaders })
  return readResponse<DocumentRecord[]>(response)
}

export async function uploadDocuments(files: File[]): Promise<{ documents: DocumentRecord[]; duplicates: DocumentRecord[] }> {
  const body = new FormData()
  files.forEach((file) => body.append('files', file))
  const response = await fetch(`${API_BASE}/upload-documents`, {
    method: 'POST',
    headers: identityHeaders,
    body,
  })
  return readResponse(response)
}

export async function deleteDocument(id: string): Promise<void> {
  const response = await fetch(`${API_BASE}/documents/${id}`, {
    method: 'DELETE',
    headers: identityHeaders,
  })
  return readResponse<void>(response)
}

export async function askQuestion(
  question: string,
  documentIds: string[],
  conversationId: string | null,
): Promise<AskResponse> {
  const response = await fetch(`${API_BASE}/ask`, {
    method: 'POST',
    headers: { ...identityHeaders, 'Content-Type': 'application/json' },
    body: JSON.stringify({
      question,
      document_ids: documentIds.length ? documentIds : null,
      conversation_id: conversationId,
      top_k: 6,
    }),
  })
  return readResponse<AskResponse>(response)
}

export async function summarizeDocument(documentId: string): Promise<AskResponse> {
  const response = await fetch(`${API_BASE}/summarize-document`, {
    method: 'POST',
    headers: { ...identityHeaders, 'Content-Type': 'application/json' },
    body: JSON.stringify({ document_id: documentId }),
  })
  return readResponse<AskResponse>(response)
}

