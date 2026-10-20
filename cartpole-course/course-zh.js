// Chinese interface adaptation; the skill's original main.js stays unchanged.
(() => {
  document.documentElement.lang = 'zh-CN';
  const translations = new Map([
    ['Exactly!', '理解正确！'], ['Not quite.', '再想一步：'],
    ['Pick an answer first!', '请先选择一个答案。'],
    ['Click "Next Step" to begin', '点击“下一步”，跟随一次交互。']
  ]);
  function translate(root) {
    const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
    let n;
    while ((n = walker.nextNode())) {
      const s = n.textContent;
      const translated = translations.get(s) || s.replace(/^(\d+ \/ \d+) messages$/, '$1 条消息').replace(/^Step (\d+) \/ (\d+)$/, '第 $1 / $2 步');
      if (s !== translated) n.textContent = translated;
    }
  }
  translate(document.body);
  new MutationObserver(records => {
    for (const r of records) {
      if (r.type === 'characterData') translate(r.target.parentNode);
      else for (const n of r.addedNodes) translate(n.nodeType === Node.TEXT_NODE ? n.parentNode : n);
    }
  }).observe(document.body, {subtree: true, childList: true, characterData: true});
  // Prevent overlapping playback timers when animation controls are clicked rapidly.
  const chat = document.querySelector('.chat-window');
  const next = chat.querySelector('.chat-next-btn');
  const all = chat.querySelector('.chat-all-btn');
  const reset = chat.querySelector('.chat-reset-btn');
  next.addEventListener('click', () => {
    next.disabled = all.disabled = reset.disabled = true;
    setTimeout(() => {next.disabled = all.disabled = reset.disabled = false;}, 850);
  });
  all.addEventListener('click', () => {
    next.disabled = all.disabled = reset.disabled = true;
    setTimeout(() => {next.disabled = all.disabled = reset.disabled = false;}, (chat.querySelectorAll('.chat-message').length + 1) * 1200);
  });
  document.querySelectorAll('.term').forEach(term => {
    term.addEventListener('focus', () => term.dispatchEvent(new Event('mouseenter')));
    term.addEventListener('blur', () => term.dispatchEvent(new Event('mouseleave')));
    term.addEventListener('keydown', event => {
      if (event.key === 'Enter' || event.key === ' ') {event.preventDefault();term.click();}
    });
  });
})();
