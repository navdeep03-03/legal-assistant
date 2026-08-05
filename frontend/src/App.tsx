import { useEffect, useMemo, useRef, useState } from 'react'
import ReactMarkdown from 'react-markdown'
import {
  AlertTriangle,
  Check,
  ChevronRight,
  FileSearch,
  FileText,
  FolderOpen,
  LoaderCircle,
  Menu,
  MessageSquareText,
  PanelRightClose,
  Paperclip,
  Plus,
  Quote,
  Scale,
  Send,
  ShieldCheck,
  Trash2,
  UploadCloud,
  X,
} from 'lucide-react'
import {
  askQuestion,
  deleteDocument,
  getDocuments,
  getHealth,
  summarizeDocument,
  uploadDocuments,
} from './api'
import type { ChatMessage, Citation, DocumentRecord, Health } from './types'

const suggestions = [
  'What are the termination rights and notice periods?',
  'Which clauses create the greatest financial risk?',
  'Summarize the confidentiality obligations.',
  'Are there any automatic renewal provisions?',
]

function App() {
  const [documents, setDocuments] = useState<DocumentRecord[]>([])
  const [selected, setSelected] = useState<Set<string>>(new Set())
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [conversationId, setConversationId] = useState<string | null>(null)
  const [question, setQuestion] = useState('')
  const [health, setHealth] = useState<Health | null>(null)
  const [activeCitation, setActiveCitation] = useState<Citation | null>(null)
  const [busy, setBusy] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const [notice, setNotice] = useState<string | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const chatFileInputRef = useRef<HTMLInputElement>(null)
  const messagesEndRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    void Promise.all([
      getDocuments().then((records) => {
        setDocuments(records)
        setSelected(new Set(records.map((item) => item.id)))
      }),
      getHealth().then(setHealth),
    ]).catch((error: Error) => setNotice(error.message))
  }, [])

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const selectedDocuments = useMemo(
    () => documents.filter((document) => selected.has(document.id)),
    [documents, selected],
  )

  const newConversation = () => {
    setMessages([])
    setConversationId(null)
    setActiveCitation(null)
    setQuestion('')
  }

  const toggleDocument = (id: string) => {
    setSelected((current) => {
      const next = new Set(current)
      if (next.has(id)) next.delete(id)
      else next.add(id)
      return next
    })
  }

  const handleUpload = async (files: File[]) => {
    if (!files.length) return
    setUploading(true)
    setNotice(null)
    try {
      const result = await uploadDocuments(files)
      const records = await getDocuments()
      setDocuments(records)
      setSelected((current) => {
        const next = new Set(current)
        result.documents.forEach((item) => next.add(item.id))
        result.duplicates.forEach((item) => next.add(item.id))
        return next
      })
      setNotice(
        result.documents.length
          ? `${result.documents.length} document${result.documents.length > 1 ? 's' : ''} indexed and ready.`
          : 'That document is already in this workspace.',
      )
    } catch (error) {
      setNotice(error instanceof Error ? error.message : 'Upload failed')
    } finally {
      setUploading(false)
      if (fileInputRef.current) fileInputRef.current.value = ''
      if (chatFileInputRef.current) chatFileInputRef.current.value = ''
    }
  }

  const handleDelete = async (document: DocumentRecord) => {
    if (!window.confirm(`Remove “${document.display_name}” and its index?`)) return
    try {
      await deleteDocument(document.id)
      setDocuments((items) => items.filter((item) => item.id !== document.id))
      setSelected((current) => {
        const next = new Set(current)
        next.delete(document.id)
        return next
      })
      if (activeCitation?.document_id === document.id) setActiveCitation(null)
    } catch (error) {
      setNotice(error instanceof Error ? error.message : 'Could not delete document')
    }
  }

  const submitQuestion = async (preset?: string) => {
    const content = (preset ?? question).trim()
    if (!content || busy) return
    setQuestion('')
    setBusy(true)
    setNotice(null)
    const userMessage: ChatMessage = { id: crypto.randomUUID(), role: 'user', content }
    const pendingId = crypto.randomUUID()
    setMessages((items) => [
      ...items,
      userMessage,
      { id: pendingId, role: 'assistant', content: '', pending: true },
    ])
    try {
      const response = await askQuestion(content, [...selected], conversationId)
      setConversationId(response.conversation_id)
      setMessages((items) =>
        items.map((item) =>
          item.id === pendingId
            ? { ...item, content: response.answer, response, pending: false }
            : item,
        ),
      )
      if (response.citations[0]) setActiveCitation(response.citations[0])
    } catch (error) {
      const message = error instanceof Error ? error.message : 'The analysis request failed.'
      setMessages((items) =>
        items.map((item) =>
          item.id === pendingId
            ? { ...item, content: `I couldn't complete that analysis: ${message}`, pending: false }
            : item,
        ),
      )
    } finally {
      setBusy(false)
    }
  }

  const handleSummary = async (document: DocumentRecord) => {
    if (busy) return
    setSidebarOpen(false)
    setBusy(true)
    const pendingId = crypto.randomUUID()
    setMessages((items) => [
      ...items,
      {
        id: crypto.randomUUID(),
        role: 'user',
        content: `Summarize ${document.display_name} and flag the material risks.`,
      },
      { id: pendingId, role: 'assistant', content: '', pending: true },
    ])
    try {
      const response = await summarizeDocument(document.id)
      setConversationId(response.conversation_id)
      setMessages((items) =>
        items.map((item) =>
          item.id === pendingId
            ? { ...item, content: response.answer, response, pending: false }
            : item,
        ),
      )
      if (response.citations[0]) setActiveCitation(response.citations[0])
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Summary failed'
      setMessages((items) =>
        items.map((item) =>
          item.id === pendingId ? { ...item, content: message, pending: false } : item,
        ),
      )
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="app-shell">
      <aside className={`sidebar ${sidebarOpen ? 'sidebar--open' : ''}`}>
        <div className="brand-row">
          <div className="brand-mark"><Scale size={20} strokeWidth={1.8} /></div>
          <div>
            <div className="brand-name">Counsel</div>
            <div className="brand-subtitle">Legal intelligence</div>
          </div>
          <button className="icon-button sidebar-close" onClick={() => setSidebarOpen(false)} aria-label="Close documents">
            <X size={18} />
          </button>
        </div>

        <button className="new-analysis" onClick={newConversation}>
          <Plus size={17} /> New analysis
        </button>

        <section className="corpus-section">
          <div className="section-heading">
            <span>Document set</span>
            <span className="count-badge">{documents.length}</span>
          </div>

          <label className={`upload-zone ${uploading ? 'is-loading' : ''}`}>
            <input
              ref={fileInputRef}
              type="file"
              accept=".pdf,.docx,.txt,.md"
              multiple
              onChange={(event) => void handleUpload(Array.from(event.target.files ?? []))}
              disabled={uploading}
            />
            {uploading ? <LoaderCircle className="spin" size={20} /> : <UploadCloud size={20} />}
            <span>{uploading ? 'Reading and indexing…' : 'Add legal documents'}</span>
            <small>PDF, DOCX, TXT · up to 20 MB</small>
          </label>

          <div className="document-toolbar">
            <span>{selected.size} selected</span>
            <button onClick={() => setSelected(new Set(selected.size === documents.length ? [] : documents.map((item) => item.id)))}>
              {selected.size === documents.length ? 'Clear' : 'Select all'}
            </button>
          </div>

          <div className="document-list">
            {documents.length === 0 ? (
              <div className="empty-documents">
                <FolderOpen size={26} strokeWidth={1.5} />
                <span>Your uploaded files will appear here.</span>
              </div>
            ) : documents.map((document) => (
              <div
                className={`document-row ${selected.has(document.id) ? 'is-selected' : ''}`}
                key={document.id}
                onClick={() => toggleDocument(document.id)}
              >
                <span className="document-check">{selected.has(document.id) && <Check size={12} strokeWidth={3} />}</span>
                <FileText size={18} className="document-icon" />
                <div className="document-copy">
                  <strong title={document.filename}>{document.display_name}</strong>
                  <span>{document.page_count ? `${document.page_count} pages` : `${document.chunk_count} sections`} · {formatBytes(document.size_bytes)}</span>
                </div>
                <div className="document-actions">
                  <button onClick={(event) => { event.stopPropagation(); void handleSummary(document) }} title="Summarize document" aria-label={`Summarize ${document.display_name}`}>
                    <FileSearch size={15} />
                  </button>
                  <button className="delete-action" onClick={(event) => { event.stopPropagation(); void handleDelete(document) }} title="Delete document" aria-label={`Delete ${document.display_name}`}>
                    <Trash2 size={14} />
                  </button>
                </div>
              </div>
            ))}
          </div>
        </section>

        <div className="privacy-note">
          <ShieldCheck size={17} />
          <div><strong>Tenant isolated</strong><span>Sources stay scoped to this workspace.</span></div>
        </div>
      </aside>

      {sidebarOpen && <button className="sidebar-scrim" onClick={() => setSidebarOpen(false)} aria-label="Close sidebar" />}

      <main className="workspace">
        <header className="workspace-header">
          <button className="icon-button mobile-menu" onClick={() => setSidebarOpen(true)} aria-label="Open documents"><Menu size={20} /></button>
          <div className="matter-heading">
            <span className="eyebrow">Matter workspace</span>
            <h1>Contract review</h1>
          </div>
          <div className="header-status">
            <span className={`status-dot ${health ? '' : 'status-dot--offline'}`} />
            <span>{health?.mode === 'openai' ? health.model : health ? 'Local demo' : 'Connecting'}</span>
          </div>
          <div className="corpus-pill"><FolderOpen size={15} /> {selectedDocuments.length || 'All'} source{selectedDocuments.length === 1 ? '' : 's'}</div>
        </header>

        <div className="conversation-layout">
          <section className="chat-panel">
            <div className="messages" aria-live="polite">
              {messages.length === 0 ? (
                <div className="welcome-state">
                  <div className="welcome-symbol"><MessageSquareText size={28} strokeWidth={1.5} /></div>
                  <span className="eyebrow">Evidence-first analysis</span>
                  <h2>Ask the agreement.<br />See the evidence.</h2>
                  <p>Get a plain-language answer grounded in your selected documents, with clause-level citations and risk flags.</p>
                  <div className="suggestion-grid">
                    {suggestions.map((suggestion) => (
                      <button key={suggestion} onClick={() => void submitQuestion(suggestion)}>
                        <span>{suggestion}</span><ChevronRight size={16} />
                      </button>
                    ))}
                  </div>
                </div>
              ) : messages.map((message) => (
                <article className={`message message--${message.role}`} key={message.id}>
                  {message.role === 'assistant' && <div className="assistant-avatar"><Scale size={15} /></div>}
                  <div className="message-content">
                    <div className="message-label">{message.role === 'user' ? 'You' : 'Counsel'}</div>
                    {message.pending ? (
                      <div className="thinking"><span /><span /><span /> Reviewing selected sources</div>
                    ) : (
                      <ReactMarkdown>{message.content}</ReactMarkdown>
                    )}
                    {message.response && (
                      <>
                        {message.response.risk_flags.length > 0 && (
                          <div className="risk-list">
                            {message.response.risk_flags.map((risk, index) => (
                              <div className={`risk-card risk-card--${risk.severity}`} key={`${risk.title}-${index}`}>
                                <AlertTriangle size={17} />
                                <div><strong>{risk.title}</strong><span>{risk.detail}</span></div>
                                <em>{risk.severity}</em>
                              </div>
                            ))}
                          </div>
                        )}
                        {message.response.citations.length > 0 && (
                          <div className="citation-strip">
                            <span className="citation-label"><Quote size={14} /> Evidence</span>
                            {message.response.citations.map((citation, index) => (
                              <button key={citation.chunk_id} onClick={() => setActiveCitation(citation)}>
                                {index + 1} · {citation.document_name}{citation.page ? ` p.${citation.page}` : ''}
                              </button>
                            ))}
                          </div>
                        )}
                        <div className="answer-meta">
                          <span className={`confidence confidence--${message.response.confidence}`}>{message.response.confidence} confidence</span>
                          <span>{message.response.mode === 'openai' ? 'Grounded synthesis' : 'Extractive demo'}</span>
                        </div>
                      </>
                    )}
                  </div>
                </article>
              ))}
              <div ref={messagesEndRef} />
            </div>

            <div className="composer-wrap">
              {notice && <div className="notice"><span>{notice}</span><button onClick={() => setNotice(null)} aria-label="Dismiss"><X size={14} /></button></div>}
              <div className="composer">
                <button
                  type="button"
                  className="attach-button"
                  onClick={() => chatFileInputRef.current?.click()}
                  disabled={uploading}
                  aria-label="Add documents"
                  title="Add PDF, DOCX, TXT, or Markdown documents"
                >
                  {uploading ? <LoaderCircle className="spin" size={18} /> : <Paperclip size={18} />}
                </button>
                <input
                  ref={chatFileInputRef}
                  className="chat-file-input"
                  type="file"
                  accept=".pdf,.docx,.txt,.md"
                  multiple
                  onChange={(event) => void handleUpload(Array.from(event.target.files ?? []))}
                  disabled={uploading}
                />
                <textarea
                  value={question}
                  onChange={(event) => setQuestion(event.target.value)}
                  onKeyDown={(event) => {
                    if (event.key === 'Enter' && !event.shiftKey) {
                      event.preventDefault()
                      void submitQuestion()
                    }
                  }}
                  rows={1}
                  placeholder={documents.length ? 'Ask about clauses, duties, dates, or risks…' : 'Upload a document, then ask a legal question…'}
                  aria-label="Legal question"
                />
                <button className="send-button" disabled={!question.trim() || busy} onClick={() => void submitQuestion()} aria-label="Send question">
                  {busy ? <LoaderCircle className="spin" size={18} /> : <Send size={18} />}
                </button>
              </div>
              <p>Answers use selected sources only. Verify important conclusions with qualified counsel.</p>
            </div>
          </section>

          <aside className={`evidence-panel ${activeCitation ? 'has-evidence' : ''}`}>
            <div className="evidence-header">
              <div><span className="eyebrow">Source inspector</span><h3>Evidence</h3></div>
              {activeCitation && <button className="icon-button" onClick={() => setActiveCitation(null)} aria-label="Close evidence"><PanelRightClose size={18} /></button>}
            </div>
            {activeCitation ? (
              <div className="evidence-content">
                <div className="source-file">
                  <div className="file-badge"><FileText size={19} /></div>
                  <div><strong>{activeCitation.document_name}</strong><span>{formatLocation(activeCitation)}</span></div>
                </div>
                <div className="source-tags">
                  <span>{activeCitation.clause_type}</span>
                  <span>{Math.round(activeCitation.relevance * 100)}% match</span>
                </div>
                <blockquote>“{activeCitation.quote}”</blockquote>
                {activeCitation.explanation && (
                  <div className="why-relevant"><strong>Why this matters</strong><p>{activeCitation.explanation}</p></div>
                )}
                <div className="source-footnote"><ShieldCheck size={15} /> Citation metadata is mapped to the retrieved chunk on the server.</div>
              </div>
            ) : (
              <div className="evidence-empty">
                <Quote size={28} strokeWidth={1.3} />
                <h3>Source details appear here</h3>
                <p>Select an evidence chip beneath an answer to inspect the quoted clause, document, page, and retrieval score.</p>
              </div>
            )}
          </aside>
        </div>
      </main>
    </div>
  )
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${Math.round(bytes / 1024)} KB`
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`
}

function formatLocation(citation: Citation): string {
  const parts = []
  if (citation.page) parts.push(`Page ${citation.page}`)
  if (citation.section) parts.push(citation.section)
  return parts.join(' · ') || 'Location not specified'
}

export default App
