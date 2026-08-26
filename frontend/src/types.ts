export type DocumentRecord = {
  id: string
  filename: string
  display_name: string
  mime_type: string
  size_bytes: number
  status: string
  page_count: number
  chunk_count: number
  created_at: string
}

export type Citation = {
  source_id: string
  document_id: string
  document_name: string
  chunk_id: string
  page: number | null
  section: string | null
  clause_type: string
  quote: string
  explanation: string
  relevance: number
}

export type RiskFlag = {
  severity: 'high' | 'medium' | 'low'
  title: string
  detail: string
}

export type AskResponse = {
  answer: string
  citations: Citation[]
  risk_flags: RiskFlag[]
  confidence: 'high' | 'medium' | 'low'
  warning: string
  source_documents: string[]
  conversation_id: string
  retrieval_used: boolean
  mode: 'mistral' | 'openai' | 'local-demo'
}

export type Health = {
  status: string
  mode: 'mistral' | 'openai' | 'local-demo'
  model: string
  embedding_model: string
  vector_engine: string
}

export type ChatMessage = {
  id: string
  role: 'user' | 'assistant'
  content: string
  response?: AskResponse
  pending?: boolean
}
