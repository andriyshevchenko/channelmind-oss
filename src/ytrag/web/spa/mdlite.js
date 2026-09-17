// Minimal, safe Markdown → HTML for chat answers (BUG-011).
// Model output is UNTRUSTED, so we ESCAPE first (no raw-HTML passthrough), then
// render a small, closed set of Markdown structures on top of the escaped text:
// paragraphs, bold, italic, inline code, http(s) links, bullet / numbered lists,
// and multi-line blockquotes. Consecutive quote lines stay in one blockquote. Nothing here can emit an attribute or tag the model
// controls.

function esc(s) {
  return String(s)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

// Inline spans, applied to already-escaped text. Links only accept http(s):// so
// a `javascript:` URL can never slip through.
function inline(s) {
  s = s.replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g,
    (_, t, u) => `<a href="${u}" target="_blank" rel="noopener" class="md-a">${t}</a>`);
  s = s.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
  s = s.replace(/(^|[^*])\*([^*\n]+)\*/g, '$1<em>$2</em>');
  s = s.replace(/`([^`]+)`/g, '<code class="md-code">$1</code>');
  return s;
}

export function renderMarkdown(src) {
  if (!src) return '';
  // Detect block-level Markdown before escaping: otherwise a leading `>` becomes
  // `&gt;` and can no longer be recognized as a quote marker. Every text fragment
  // still goes through esc() before inline rendering below.
  const lines = String(src).split(/\r?\n/);
  const out = [];
  let list = null;      // 'ul' | 'ol' | null
  let para = [];
  let quoteLines = [];
  const flushPara = () => {
    if (para.length) { out.push('<p class="md-p">' + para.map(inline).join('<br>') + '</p>'); para = []; }
  };
  const flushList = () => { if (list) { out.push('</' + list + '>'); list = null; } };
  const flushQuote = () => {
    if (quoteLines.length) {
      out.push('<blockquote class="md-quote">' + quoteLines.map(inline).join('<br>') + '</blockquote>');
      quoteLines = [];
    }
  };
  for (const raw of lines) {
    const line = raw.replace(/\s+$/, '');
    const bullet = line.match(/^\s*[-•]\s+(.*)$/);
    const numbered = line.match(/^\s*\d+\.\s+(.*)$/);
    const quote = line.match(/^\s*>\s?(.*)$/);
    if (quote) {
      flushPara(); flushList();
      quoteLines.push(esc(quote[1]));
    } else if (bullet) {
      flushPara(); flushQuote();
      if (list !== 'ul') { flushList(); out.push('<ul class="md-ul">'); list = 'ul'; }
      out.push('<li>' + inline(esc(bullet[1])) + '</li>');
    } else if (numbered) {
      flushPara(); flushQuote();
      if (list !== 'ol') { flushList(); out.push('<ol class="md-ol">'); list = 'ol'; }
      out.push('<li>' + inline(esc(numbered[1])) + '</li>');
    } else if (!line.trim()) {
      flushPara(); flushList(); flushQuote();
    } else {
      flushList(); flushQuote();
      para.push(esc(line));
    }
  }
  flushPara(); flushList(); flushQuote();
  return out.join('');
}
