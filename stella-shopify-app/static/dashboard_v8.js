/* ═══════════════════════════════════════════════════════════════
   STELLA OS — Dashboard JavaScript
   Tour de controle IA PlaneteBeauty
   Refonte SaaS — 07/04/2026
   ═══════════════════════════════════════════════════════════════ */

const STELLA = {
  state: { activeTab: 'home', loaded: {} },

  // ═══ INIT ═══
  init() {
    this.initTabs();
    this.updateClock();
    setInterval(() => this.updateClock(), 30000);
    this.home.load();
    setInterval(() => { if (this.state.activeTab === 'home') this.home.load(); }, 300000);
  },

  updateClock() {
    const el = document.getElementById('header-time');
    if (el) el.textContent = new Date().toLocaleTimeString('fr-FR', { hour: '2-digit', minute: '2-digit', timeZone: 'Europe/Paris' });
  },

  // ═══ TABS ═══
  initTabs() {
    document.querySelectorAll('.tab-btn').forEach(btn => {
      btn.addEventListener('click', () => this.switchTab(btn.dataset.tab));
    });
  },

  switchTab(tabId) {
    this.state.activeTab = tabId;
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.toggle('active', b.dataset.tab === tabId));
    document.querySelectorAll('.tab-panel').forEach(p => p.classList.toggle('active', p.id === 'panel-' + tabId));
    if (!this.state.loaded[tabId]) {
      this.state.loaded[tabId] = true;
      const loader = {home:'home',cashback:'cashback',tryme:'tryme',reviews:'reviews',catalogue:'catalogue',
        bis:'bis',gads:'gads',promo:'promo',quiz:'quiz',ops:'ops',shipping:'shipping',system:'system',stellamem:'stellamem'};
      if (loader[tabId] && this[loader[tabId]] && this[loader[tabId]].load) this[loader[tabId]].load();
    }
  },

  refreshAll() {
    this.state.loaded = {};
    const tab = this.state.activeTab;
    if (this[tab] && this[tab].load) this[tab].load();
    this.showToast('Donnees rafraichies');
  },

  // ═══ API ═══
  async api(url) {
    try {
      const r = await fetch(url, { headers: { 'X-API-Key': 'stella-mem-2026-planetebeauty' } });
      if (!r.ok) throw new Error(r.status);
      return await r.json();
    } catch (e) { console.error('API', url, e); return null; }
  },

  async apiPost(url, body) {
    try {
      const r = await fetch(url, { method: 'POST', headers: { 'Content-Type': 'application/json', 'X-API-Key': 'stella-mem-2026-planetebeauty' }, body: JSON.stringify(body) });
      return await r.json();
    } catch (e) { console.error('API POST', url, e); return null; }
  },

  // ═══ UTILS ═══
  eur(n) { return (n || 0).toLocaleString('fr-FR', { style: 'currency', currency: 'EUR' }); },
  shortTime(iso) { if (!iso) return '--'; return new Date(iso).toLocaleDateString('fr-FR', { day: '2-digit', month: '2-digit', hour: '2-digit', minute: '2-digit', timeZone: 'Europe/Paris' }); },
  relativeTime(iso) {
    if (!iso) return '--';
    const diff = (Date.now() - new Date(iso).getTime()) / 1000;
    if (diff < 60) return 'a l\'instant';
    if (diff < 3600) return Math.floor(diff / 60) + ' min';
    if (diff < 86400) return Math.floor(diff / 3600) + ' h';
    return Math.floor(diff / 86400) + ' j';
  },
  statusBadge(status) {
    const map = { PAID: 'success', PARTIALLY_PAID: 'warning', UNFULFILLED: 'warning', FULFILLED: 'success', active: 'success', pending: 'warning', used: 'gold', expired: 'error', notified: 'info', completed: 'success', failed: 'error' };
    return `<span class="badge-pill badge-${map[status] || 'info'}">${status}</span>`;
  },
  showToast(msg, type = 'success') {
    const t = document.createElement('div');
    t.className = `toast ${type}`;
    t.textContent = msg;
    document.body.appendChild(t);
    setTimeout(() => t.remove(), 3000);
  },
  showModal(title, bodyHTML, footerHTML) {
    document.getElementById('modal-title').textContent = title;
    document.getElementById('modal-body').innerHTML = bodyHTML;
    document.getElementById('modal-footer').innerHTML = footerHTML || '';
    document.getElementById('modal-overlay').style.display = 'flex';
  },
  closeModal() { document.getElementById('modal-overlay').style.display = 'none'; },

  // ═══════════════════════════════════════════════════════════════
  // VUE 1: DASHBOARD
  // ═══════════════════════════════════════════════════════════════
  home: {
    logFilter: 'business',
    async load() {
      const [kpis, activity, status, ops] = await Promise.all([
        STELLA.api('/api/kpis/summary'),
        STELLA.api('/api/activity/log?limit=30'),
        STELLA.api('/api/system/status'),
        STELLA.api('/api/ops/briefing')
      ]);

      // KPIs STELLA
      if (kpis) {
        document.getElementById('kpi-cashback-encours').textContent = STELLA.eur(kpis.cashback_active_total || 0);
        document.getElementById('kpi-tryme-pending').textContent = kpis.tryme_active || 0;
        document.getElementById('kpi-reviews-pending').textContent = kpis.reviews_pending || 0;
        document.getElementById('kpi-bis-active').textContent = kpis.bis_active || 0;
        document.getElementById('kpi-catalogue-issues').textContent = kpis.catalogue_issues || 0;
        document.getElementById('kpi-quiz-completions').textContent = kpis.quiz_completions_today || 0;
        // Badges
        const bRev = document.getElementById('badge-reviews');
        if (bRev) bRev.textContent = kpis.reviews_pending || 0;
        const bBis = document.getElementById('badge-bis');
        if (bBis) bBis.textContent = kpis.bis_active || 0;
      }

      // Alertes STELLA OS
      if (ops && ops.alerts_24h && ops.alerts_24h.length > 0) {
        const bar = document.getElementById('alerts-bar');
        bar.style.display = 'block';
        bar.innerHTML = ops.alerts_24h.map(a => {
          const cls = a.severity === 'critical' ? 'alert-critical' : a.severity === 'error' ? 'alert-critical' : 'alert-warning';
          return `<div class="alert-item ${cls}">${a.action}</div>`;
        }).join('');
      }

      // Badge ops
      if (ops) {
        const bOps = document.getElementById('badge-ops');
        const count = (ops.active_operations || []).length + (ops.upcoming_actions_48h || []).length;
        if (bOps) bOps.textContent = count || '';
      }

      // Suggestions STELLA (intelligence)
      const suggestions = await STELLA.api('/api/ops/suggestions');
      const qa = document.getElementById('quick-actions');
      if (qa && suggestions && suggestions.suggestions && suggestions.suggestions.length > 0) {
        const priorityIcon = { critical: '🔴', high: '🟠', medium: '🟡', low: '🔵' };
        qa.innerHTML = suggestions.suggestions.map(s =>
          `<button class="quick-action-btn" onclick="STELLA.switchTab('${s.action_tab}')">
            ${priorityIcon[s.priority] || '⚪'} ${s.message}
          </button>`
        ).join('');
      } else if (qa) {
        qa.innerHTML = '<span style="font-size:0.8rem;color:var(--success)">&#10003; Aucune action requise — tout est sous controle</span>';
      }

      // Global status
      if (status && status.services) {
        const allOk = Object.values(status.services).every(s => s === 'online');
        const dot = document.getElementById('global-status');
        if (dot) dot.className = `status-dot ${allOk ? 'online' : 'offline'}`;
      }

      // Activity feed
      this.renderLog(activity);
    },

    filterLog(filter) {
      this.logFilter = filter;
      document.querySelectorAll('.filter-btn').forEach(b => b.classList.toggle('active', b.dataset.filter === filter));
      STELLA.api('/api/activity/log?limit=30').then(d => this.renderLog(d));
    },

    renderLog(data) {
      const feed = document.getElementById('activity-feed');
      if (!data || !data.recent) { feed.innerHTML = '<div class="empty-state"><p>Aucune activite</p></div>'; return; }
      const CRON_TYPES = ['cron_run', 'cron'];
      const items = this.logFilter === 'business'
        ? data.recent.filter(a => !CRON_TYPES.includes(a.type) && !(a.action || '').startsWith('Cron '))
        : data.recent;
      const icons = { cashback_credited: '💰', tryme_created: '🧪', order_paid: '🛒', review_submitted: '⭐', bis_notified: '📧', product_change: '📦', quiz_regen: '🎯', error: '🔴' };
      feed.innerHTML = items.slice(0, 20).map(a => `
        <div class="activity-item">
          <span class="activity-icon">${icons[a.type] || '⚪'}</span>
          <div class="activity-content">
            <div class="activity-action">${a.action || ''}</div>
            ${a.details ? `<div class="activity-detail">${typeof a.details === 'string' ? a.details : (a.details.product_title || a.details.customer_email || '')}</div>` : ''}
          </div>
          <span class="activity-time">${STELLA.relativeTime(a.timestamp)}</span>
        </div>`).join('') || '<div class="empty-state"><p>Aucune activite</p></div>';
    }
  },

  // ═══════════════════════════════════════════════════════════════
  // VUE 2: CASHBACK
  // ═══════════════════════════════════════════════════════════════
  cashback: {
    async load() {
      const [data, settings] = await Promise.all([
        STELLA.api('/api/cashback/dashboard'),
        STELLA.api('/api/cashback/settings')
      ]);
      if (!data) return;

      // Health
      const health = document.getElementById('cb-health');
      if (health) health.innerHTML = `
        <div class="kpi-card"><div class="kpi-value" style="font-size:1.1rem">${data.total_rewarded || 0}</div><div class="kpi-label">Credits generes</div></div>
        <div class="kpi-card"><div class="kpi-value gold" style="font-size:1.1rem">${STELLA.eur(data.total_amount || 0)}</div><div class="kpi-label">Montant total</div></div>`;

      // KPIs
      const pendingList = data.pending || [];
      const pendingAmount = pendingList.reduce((s, r) => s + (parseFloat(r.cashback_amount) || 0), 0);
      const kpis = document.getElementById('cb-kpis');
      if (kpis) kpis.innerHTML = `
        <div class="kpi-card"><div class="kpi-value">${pendingList.length}</div><div class="kpi-label">Encours actifs</div></div>
        <div class="kpi-card"><div class="kpi-value gold">${STELLA.eur(pendingAmount)}</div><div class="kpi-label">Encours EUR</div></div>
        <div class="kpi-card"><div class="kpi-value">${data.total_used || 0}</div><div class="kpi-label">Utilises</div></div>
        <div class="kpi-card"><div class="kpi-value">${data.total_revoked || 0}</div><div class="kpi-label">Revoques</div></div>`;

      // Expiring
      const expCount = document.getElementById('cb-expiring-count');
      if (expCount) expCount.textContent = (data.expiring_soon || []).length;
      const expTb = document.querySelector('#cb-expiring-table tbody');
      if (expTb) expTb.innerHTML = (data.expiring_soon || []).map(r => `<tr><td>${r.customer_email || ''}</td><td>${r.order_name || ''}</td><td style="font-weight:bold">${STELLA.eur(r.cashback_amount)}</td><td>${STELLA.shortTime(r.expires_at)}</td><td>${r.reminder_sent_at ? '⏰' : '✉️'}</td></tr>`).join('') || '<tr><td colspan="5" class="empty-state">Aucun cashback a expiration proche</td></tr>';

      // Pending
      const pendCount = document.getElementById('cb-pending-count');
      if (pendCount) pendCount.textContent = pendingList.length;
      const pendTb = document.querySelector('#cb-pending-table tbody');
      if (pendTb) pendTb.innerHTML = pendingList.slice(0, 50).map(r => `<tr><td>${r.customer_email || ''}</td><td>${r.order_name || ''}</td><td style="font-weight:bold">${STELLA.eur(r.cashback_amount)}</td><td>${STELLA.shortTime(r.expires_at)}</td></tr>`).join('') || '<tr><td colspan="4" class="empty-state">Aucun encours</td></tr>';

      // Settings
      if (settings) {
        document.getElementById('cb-rate').value = (settings.cashback_rate || 0.05) * 100;
        document.getElementById('cb-expiry').value = settings.expiry_days || 60;
        document.getElementById('cb-min-use').value = settings.min_order_use || 70;
        document.getElementById('cb-min-amount').value = settings.min_cashback_amount || 0.5;
        document.getElementById('cb-excluded-tags').value = settings.excluded_tags || 'tryme,no-cashback';
      }
    },
    async save() {
      const rate = parseFloat(document.getElementById('cb-rate').value);
      const expiry = parseInt(document.getElementById('cb-expiry').value);
      if (isNaN(rate) || rate <= 0 || rate > 100) { STELLA.showToast('Taux invalide (1-100%)', 'error'); return; }
      if (isNaN(expiry) || expiry <= 0) { STELLA.showToast('Expiration invalide', 'error'); return; }
      const result = await STELLA.apiPost('/api/cashback/settings', {
        cashback_rate: rate / 100,
        expiry_days: expiry,
        min_order_use: parseFloat(document.getElementById('cb-min-use').value) || 70,
        min_cashback_amount: parseFloat(document.getElementById('cb-min-amount').value) || 0.5,
        excluded_tags: document.getElementById('cb-excluded-tags').value
      });
      if (result) STELLA.showToast('Parametres cashback sauvegardes');
    }
  },

  // ═══════════════════════════════════════════════════════════════
  // VUE 3: TRY ME
  // ═══════════════════════════════════════════════════════════════
  tryme: {
    async load() {
      const data = await STELLA.api('/api/tryme/dashboard');
      if (!data) return;

      // Health
      const health = document.getElementById('tryme-health');
      if (health) health.innerHTML = `
        <div class="kpi-card"><div class="kpi-value">${data.total_codes || 0}</div><div class="kpi-label">Codes generes</div></div>
        <div class="kpi-card"><div class="kpi-value">${data.active_codes || 0}</div><div class="kpi-label">Actifs</div></div>`;

      // KPIs
      const kpis = document.getElementById('tryme-kpis');
      if (kpis) kpis.innerHTML = `
        <div class="kpi-card"><div class="kpi-value gold">${data.used_codes || 0}</div><div class="kpi-label">Convertis</div></div>
        <div class="kpi-card"><div class="kpi-value">${data.expired_codes || 0}</div><div class="kpi-label">Expires</div></div>
        <div class="kpi-card"><div class="kpi-value">${data.total_codes > 0 ? Math.round((data.used_codes || 0) / data.total_codes * 100) : 0}%</div><div class="kpi-label">Taux conversion</div></div>`;

      // Codes table
      const tbody = document.getElementById('tryme-codes-tbody');
      if (tbody) tbody.innerHTML = (data.recent_codes || data.codes || []).map(c => {
        const statusClass = c.status === 'used' ? 'gold' : c.status === 'expired' ? 'error' : 'warning';
        const isUpsell = c.tryme_type === 'upsell' || (c.discount_code || '').startsWith('TU-');
        const typeLabel = isUpsell ? '<span class="badge-pill badge-gold" style="font-size:0.6rem">UPSELL -5%</span>' : '';
        const priceDisplay = isUpsell ? '-5%' : STELLA.eur(c.tryme_price || c.amount || 0);
        return `<tr${isUpsell ? ' style="background:rgba(196,149,106,0.05)"' : ''}>
          <td>${c.order_name || ''}</td><td style="font-weight:600">${c.discount_code || ''} ${typeLabel}</td>
          <td>${c.product_title || ''}</td><td>${priceDisplay}</td>
          <td>${c.customer_email || ''}</td><td>${STELLA.shortTime(c.discount_expires_at || c.expires_at)}</td>
          <td><span class="badge-pill badge-${statusClass}">${c.status}</span></td>
          <td>${!isUpsell && c.order_id ? `<a href="/api/tryme/card-pdf/${c.order_id}" target="_blank" class="btn-outline btn-sm">PDF</a>` : ''}</td>
        </tr>`;
      }).join('') || '<tr><td colspan="8" class="empty-state">Aucun code</td></tr>';

    }
  },

  // ═══════════════════════════════════════════════════════════════
  // VUE 4: AVIS
  // ═══════════════════════════════════════════════════════════════
  reviews: {
    async load() {
      const data = await STELLA.api('/api/reviews/dashboard');
      if (!data) return;

      // Health — pending count from separate endpoint
      const pendingData = await STELLA.api('/api/reviews/pending');
      const pendingCount = pendingData && pendingData.reviews ? pendingData.reviews.length : 0;
      const sourceCount = data.by_source ? Object.keys(data.by_source).length : 0;
      const health = document.getElementById('reviews-health');
      if (health) health.innerHTML = `
        <div class="kpi-card"><div class="kpi-value">${pendingCount}</div><div class="kpi-label">En attente</div></div>
        <div class="kpi-card"><div class="kpi-value">${sourceCount}</div><div class="kpi-label">Sources</div></div>`;

      // KPIs
      const kpis = document.getElementById('reviews-kpis');
      if (kpis) kpis.innerHTML = `
        <div class="kpi-card"><div class="kpi-value gold">${data.total || 0}</div><div class="kpi-label">Total avis</div></div>
        <div class="kpi-card"><div class="kpi-value">${(data.avg_rating || 0).toFixed(1)}/5</div><div class="kpi-label">Note moyenne</div></div>`;

      // Recent table
      const tbody = document.querySelector('#reviews-table tbody');
      if (tbody) tbody.innerHTML = (data.recent || []).slice(0, 20).map(r => `<tr>
        <td>${r.product_handle || ''}</td>
        <td>${'\u2B50'.repeat(Math.round(r.rating || 0))}</td>
        <td><span class="badge-pill badge-${r.source === 'flow-email' ? 'gold' : 'info'}">${r.source === 'flow-email' ? 'Email' : 'Site'}</span></td>
        <td>${STELLA.shortTime(r.created_at)}</td>
      </tr>`).join('') || '<tr><td colspan="4" class="empty-state">Aucun avis</td></tr>';

      this.loadPending();
    },
    async loadPending() {
      const data = await STELLA.api('/api/reviews/pending');
      const list = document.getElementById('reviews-pending-list');
      if (!data || !data.reviews || data.reviews.length === 0) {
        list.innerHTML = '<div class="empty-state"><p>Aucun avis en attente</p></div>';
        return;
      }
      list.innerHTML = data.reviews.map(r => `
        <div class="review-card">
          <div class="review-card-header">
            <span>${'\u2B50'.repeat(Math.round(r.rating))} <strong>${r.reviewer_name || 'Anonyme'}</strong></span>
            <span class="badge-pill badge-${r.source === 'flow-email' ? 'gold' : 'info'}">${r.source === 'flow-email' ? 'Via email' : 'Page produit'}</span>
          </div>
          <div class="review-card-meta">${r.product_handle || ''} — ${STELLA.shortTime(r.created_at)}</div>
          <div class="review-card-body">${r.title ? `<strong>${r.title}</strong><br>` : ''}${r.body || ''}</div>
          <div class="review-card-actions">
            ${r.source === 'flow-email'
              ? `<button class="btn-success" onclick="STELLA.reviews.approve(${r.id}, true)">Approuver + 5€ credit</button>`
              : `<button class="btn-success" onclick="STELLA.reviews.approve(${r.id}, false)">Approuver</button>`}
            <button class="btn-danger" onclick="STELLA.reviews.reject(${r.id})">Rejeter</button>
          </div>
        </div>`).join('');
    },
    async approve(id, withCredit) {
      if (!confirm('Approuver cet avis ?')) return;
      await STELLA.apiPost('/api/reviews/approve', { review_id: id, with_credit: withCredit });
      STELLA.showToast('Avis approuve' + (withCredit ? ' + 5€ credit' : ''));
      this.loadPending();
    },
    async reject(id) {
      if (!confirm('Rejeter cet avis ? Cette action est irreversible.')) return;
      await STELLA.apiPost('/api/reviews/reject', { review_id: id });
      STELLA.showToast('Avis rejete');
      this.loadPending();
    }
  },

  // ═══════════════════════════════════════════════════════════════
  // VUE 5: CATALOGUE
  // ═══════════════════════════════════════════════════════════════
  catalogue: {
    async load() {
      const data = await STELLA.api('/api/catalogue/dashboard');
      if (!data) return;
      const kpis = document.getElementById('catalogue-kpis');
      if (kpis) kpis.innerHTML = `
        <div class="kpi-card"><div class="kpi-value">${data.total_products || 0}</div><div class="kpi-label">Total produits</div></div>
        <div class="kpi-card"><div class="kpi-value" style="color:${data.products_with_issues > 0 ? 'var(--warning)' : 'var(--success)'}">${data.products_with_issues || 0}</div><div class="kpi-label">Avec problemes</div></div>
        <div class="kpi-card"><div class="kpi-value">${STELLA.relativeTime(data.last_audit)}</div><div class="kpi-label">Dernier audit</div></div>`;
      const tbody = document.querySelector('#catalogue-issues tbody');
      if (tbody) {
        const issues = data.issues || [];
        tbody.innerHTML = issues.map(i => {
          const probs = (i.problems || i.issues || []);
          const severity = probs.some(p => p.includes('SEO')) ? 'error' : 'warning';
          return `<tr><td>${i.title || ''}</td><td><span class="badge-pill badge-${severity}">${severity === 'error' ? 'Critique' : 'Warning'}</span></td><td>${probs.join(', ')}</td></tr>`;
        }).join('') || '<tr><td colspan="3" class="empty-state">Aucun probleme detecte</td></tr>';
      }
    },
    async forceAudit() {
      STELLA.showToast('Audit en cours...');
      await STELLA.apiPost('/api/cron/audit-qualite', {});
      STELLA.showToast('Audit termine');
      this.load();
    },
    async syncTags() {
      STELLA.showToast('Sync tags en cours...');
      await STELLA.apiPost('/api/cron/sync-tags', {});
      STELLA.showToast('Tags synchronises');
    }
  },

  // ═══════════════════════════════════════════════════════════════
  // VUE 6: BACK IN STOCK
  // ═══════════════════════════════════════════════════════════════
  bis: {
    async load() {
      const [data, smtp] = await Promise.all([
        STELLA.api('/api/bis/dashboard'),
        STELLA.api('/api/bis/smtp-status')
      ]);
      if (!data) return;
      const kpis = document.getElementById('bis-kpis');
      if (kpis) kpis.innerHTML = `
        <div class="kpi-card"><div class="kpi-value">${data.total_active || 0}</div><div class="kpi-label">Abonnes actifs</div></div>
        <div class="kpi-card"><div class="kpi-value">${data.total_products || 0}</div><div class="kpi-label">Produits surveilles</div></div>
        <div class="kpi-card"><div class="kpi-value">${data.total_notified || 0}</div><div class="kpi-label">Emails envoyes</div></div>
        <div class="kpi-card"><div class="kpi-value">${data.total_subscriptions || 0}</div><div class="kpi-label">Total inscriptions</div></div>`;
      const smtpBadge = document.getElementById('bis-smtp-status');
      if (smtpBadge && smtp) {
        smtpBadge.textContent = `SMTP: ${smtp.status || 'unknown'}`;
        smtpBadge.className = `badge-pill badge-${smtp.status === 'ok' ? 'success' : 'error'}`;
      }
      // Subscriptions table
      const subs = data.recent || data.subscriptions || [];
      const tbody = document.querySelector('#bis-table tbody');
      if (tbody) tbody.innerHTML = subs.slice(0, 20).map(s => `<tr>
        <td>${s.email || ''}</td><td>${s.product_title || s.product_handle || ''}</td>
        <td>${STELLA.shortTime(s.created_at)}</td>
        <td>${STELLA.statusBadge(s.status)}</td>
      </tr>`).join('') || '<tr><td colspan="4" class="empty-state">Aucun abonne</td></tr>';
    }
  },

  // ═══════════════════════════════════════════════════════════════
  // VUE 7: MARKETING / CODES PROMO
  // ═══════════════════════════════════════════════════════════════
  promo: {
    _config: null,
    async load() {
      const data = await STELLA.api('/api/promo/settings');
      if (!data) return;
      this._config = data;
      this.render();
      // History from ops
      const ops = await STELLA.api('/api/ops/audit?entity=bhtc_fr&limit=20');
      const hist = document.getElementById('promo-history');
      if (hist && ops) {
        const promoOps = (ops.audit_log || []).filter(a => a.action && (a.action.includes('promo') || a.action.includes('operation')));
        hist.innerHTML = promoOps.length > 0
          ? promoOps.map(o => `<div class="activity-item"><span class="activity-time">${STELLA.shortTime(o.timestamp)}</span><div class="activity-content"><div class="activity-action">${o.action}</div></div></div>`).join('')
          : '<div class="empty-state"><p>Aucune operation passee</p></div>';
      }
    },
    render() {
      if (!this._config) return;
      const codes = this._config.codes || {};
      const tbody = document.getElementById('promo-tbody');
      if (tbody) tbody.innerHTML = Object.entries(codes).map(([name, c]) => `<tr>
        <td style="font-weight:700;color:var(--gold)">${name}</td>
        <td>-${c.percent}%</td><td>${c.minSubtotal}€+</td><td>--</td>
        <td><button class="btn-outline btn-sm" onclick="STELLA.promo.editCode('${name}')">Modifier</button> <button class="btn-outline btn-sm" style="color:var(--error)" onclick="STELLA.promo.deleteCode('${name}')">Suppr</button></td>
      </tr>`).join('') || '<tr><td colspan="5" class="empty-state">Aucun code</td></tr>';
      const vendors = document.getElementById('promo-excluded-vendors');
      if (vendors) vendors.value = (this._config.excludedVendors || []).join(', ');
    },
    showAddModal() {
      STELLA.showModal('Ajouter un code promo',
        `<div class="form-group"><label class="form-label">Nom du code</label><input class="form-input" id="modal-code-name" placeholder="PB580"></div>
         <div class="form-row"><div class="form-group"><label class="form-label">Reduction (%)</label><input class="form-input" id="modal-code-percent" type="number" min="1" max="100" placeholder="5"></div>
         <div class="form-group"><label class="form-label">Panier minimum (EUR)</label><input class="form-input" id="modal-code-min" type="number" min="0" placeholder="80"></div></div>`,
        `<button class="btn-outline" onclick="STELLA.closeModal()">Annuler</button><button class="btn-gold" onclick="STELLA.promo.addCode()">Ajouter</button>`);
    },
    addCode() {
      const name = (document.getElementById('modal-code-name').value || '').trim().toUpperCase();
      const pct = parseFloat(document.getElementById('modal-code-percent').value);
      const min = parseFloat(document.getElementById('modal-code-min').value) || 0;
      if (!name || isNaN(pct) || pct <= 0) { STELLA.showToast('Champs invalides', 'error'); return; }
      if (!this._config) this._config = { codes: {}, excludedVendors: [] };
      this._config.codes[name] = { percent: pct, minSubtotal: min, message: `-${pct}% avec le code ${name}` };
      this.render();
      STELLA.closeModal();
      STELLA.showToast(`Code ${name} ajoute`);
    },
    editCode(name) {
      const c = this._config.codes[name];
      if (!c) return;
      STELLA.showModal(`Modifier ${name}`,
        `<div class="form-row"><div class="form-group"><label class="form-label">Reduction (%)</label><input class="form-input" id="modal-edit-percent" type="number" value="${c.percent}"></div>
         <div class="form-group"><label class="form-label">Panier minimum (EUR)</label><input class="form-input" id="modal-edit-min" type="number" value="${c.minSubtotal}"></div></div>`,
        `<button class="btn-outline" onclick="STELLA.closeModal()">Annuler</button><button class="btn-gold" onclick="STELLA.promo.applyEdit('${name}')">Enregistrer</button>`);
    },
    applyEdit(name) {
      const pct = parseFloat(document.getElementById('modal-edit-percent').value);
      const min = parseFloat(document.getElementById('modal-edit-min').value) || 0;
      if (isNaN(pct) || pct <= 0) { STELLA.showToast('Pourcentage invalide', 'error'); return; }
      this._config.codes[name] = { percent: pct, minSubtotal: min, message: `-${pct}% avec le code ${name}` };
      this.render();
      STELLA.closeModal();
    },
    deleteCode(name) {
      if (!confirm(`Supprimer le code ${name} ?`)) return;
      delete this._config.codes[name];
      this.render();
      STELLA.showToast(`Code ${name} supprime`);
    },
    async save() {
      if (!this._config) return;
      this._config.excludedVendors = document.getElementById('promo-excluded-vendors').value.split(',').map(v => v.trim()).filter(v => v);
      await STELLA.apiPost('/api/promo/settings', this._config);
      STELLA.showToast('Codes promo sauvegardes');
    }
  },

  // ═══════════════════════════════════════════════════════════════
  // VUE 8: QUIZ
  // ═══════════════════════════════════════════════════════════════
  quiz: {
    async load() {
      const data = await STELLA.api('/api/quiz/stats');
      if (!data) return;
      const today = data.today || {};
      const kpis = document.getElementById('quiz-kpis');
      if (kpis) kpis.innerHTML = `
        <div class="kpi-card"><div class="kpi-value">${today.quiz_view || 0}</div><div class="kpi-label">Vues</div></div>
        <div class="kpi-card"><div class="kpi-value">${today.quiz_complete || 0}</div><div class="kpi-label">Completions</div></div>
        <div class="kpi-card"><div class="kpi-value gold">${today.quiz_add_to_cart || 0}</div><div class="kpi-label">Ajouts panier</div></div>
        <div class="kpi-card"><div class="kpi-value">${data.products_count || 0}</div><div class="kpi-label">Produits indexes</div></div>`;
      const det = document.getElementById('quiz-details');
      if (det) det.innerHTML = `<p style="font-size:0.85rem;color:var(--text-secondary)">Derniere regeneration : ${STELLA.relativeTime(data.last_regenerated)}<br>Taux completion : ${today.quiz_view > 0 ? Math.round((today.quiz_complete || 0) / today.quiz_view * 100) : 0}%</p>`;
    },
    async regenerate() {
      STELLA.showToast('Regeneration en cours...');
      await STELLA.apiPost('/api/quiz/regenerate', {});
      STELLA.showToast('quiz-data.json regenere');
      this.load();
    }
  },

  // ═══════════════════════════════════════════════════════════════
  // VUE 9: STELLA OS
  // ═══════════════════════════════════════════════════════════════
  ops: {
    async load() {
      const [briefing, health] = await Promise.all([
        STELLA.api('/api/ops/briefing'),
        STELLA.api('/api/ops/health')
      ]);
      if (!briefing) return;

      // KPIs
      const kpis = document.getElementById('ops-kpis');
      if (kpis) kpis.innerHTML = `
        <div class="kpi-card"><div class="kpi-value">${(briefing.active_operations || []).length}</div><div class="kpi-label">Ops actives</div></div>
        <div class="kpi-card"><div class="kpi-value">${(briefing.upcoming_actions_48h || []).length}</div><div class="kpi-label">Actions 48h</div></div>
        <div class="kpi-card"><div class="kpi-value">${(briefing.alerts_24h || []).length}</div><div class="kpi-label">Alertes 24h</div></div>
        <div class="kpi-card"><div class="kpi-value">${health ? health.pending_actions : 0}</div><div class="kpi-label">Rollbacks en attente</div></div>`;

      // Active operations
      const activeList = document.getElementById('ops-active-list');
      if (activeList) {
        const ops = briefing.active_operations || [];
        activeList.innerHTML = ops.length > 0 ? ops.map(o => `
          <div class="ops-card">
            <div class="ops-card-info">
              <div class="ops-card-name">${o.name}</div>
              <div class="ops-card-meta">${o.type} — cree ${STELLA.shortTime(o.created_at)}</div>
            </div>
            ${o.expires_at ? `<div class="ops-card-countdown">Expire ${STELLA.relativeTime(o.expires_at)}</div>` : ''}
          </div>`).join('') : '<div class="empty-state"><p>Aucune operation active</p></div>';
      }

      // Site state
      const stateTb = document.querySelector('#ops-state-table tbody');
      if (stateTb) {
        const states = briefing.site_state || [];
        stateTb.innerHTML = states.map(s => `<tr>
          <td><span class="badge-pill badge-info">${s.category}</span></td>
          <td style="font-weight:600">${s.key}</td>
          <td style="font-size:0.75rem;color:var(--text-muted)">${typeof s.value === 'object' ? JSON.stringify(s.value).substring(0, 80) : s.value}</td>
          <td>${s.expires_at ? STELLA.shortTime(s.expires_at) : '--'}</td>
        </tr>`).join('') || '<tr><td colspan="4" class="empty-state">Aucun element</td></tr>';
      }

      // Scheduled
      const schedList = document.getElementById('ops-scheduled-list');
      const upcoming = briefing.upcoming_actions_48h || [];
      if (schedList) schedList.innerHTML = upcoming.length > 0 ? upcoming.map(u => `
        <div class="ops-card">
          <div class="ops-card-info">
            <div class="ops-card-name">${u.operation_name || u.action_type}</div>
            <div class="ops-card-meta">${u.action_type} — planifie ${STELLA.shortTime(u.scheduled_at)}</div>
          </div>
          <span class="badge-pill badge-warning">${u.status}</span>
        </div>`).join('') : '<div class="empty-state"><p>Aucune action en attente</p></div>';

      // Audit trail
      const auditTb = document.querySelector('#ops-audit-table tbody');
      const alerts = briefing.alerts_24h || [];
      if (auditTb) auditTb.innerHTML = alerts.map(a => `<tr>
        <td>${STELLA.shortTime(a.timestamp)}</td><td>${a.actor || ''}</td>
        <td>${a.action}</td><td>${STELLA.statusBadge(a.severity)}</td>
      </tr>`).join('') || '<tr><td colspan="4" class="empty-state">Aucun evenement</td></tr>';
    }
  },

  // ═══════════════════════════════════════════════════════════════
  // VUE 10: LIVRAISON
  // ═══════════════════════════════════════════════════════════════
  shipping: {
    async load() {
      const data = await STELLA.api('/api/shipping/settings');
      if (!data) return;
      document.getElementById('shipping-threshold').value = data.threshold || 99;
      document.getElementById('shipping-amount').value = data.amount || 5;
      const status = document.getElementById('shipping-status');
      if (status) status.textContent = `ACTIF -${data.amount || 5}€ des ${data.threshold || 99}€`;
      // Calculer "apres reduction"
      const amt = data.amount || 5;
      const rates = [
        { relay: 8, home: 12 },
        { relay: 8, home: 12 },
        { relay: null, home: 12 },
        { relay: null, home: 11 },
        { relay: null, home: 15 }
      ];
      document.querySelectorAll('.ship-after').forEach((el, i) => {
        const r = rates[i];
        const relayAfter = r.relay ? Math.max(0, r.relay - amt) : null;
        const homeAfter = Math.max(0, r.home - amt);
        const parts = [];
        if (relayAfter !== null) parts.push(`Relay ${relayAfter}€`);
        parts.push(`Dom ${homeAfter}€`);
        el.textContent = parts.join(' / ');
      });
    },
    async save() {
      const threshold = parseFloat(document.getElementById('shipping-threshold').value);
      const amount = parseFloat(document.getElementById('shipping-amount').value);
      if (!threshold || !amount || threshold <= 0 || amount <= 0) { STELLA.showToast('Valeurs invalides', 'error'); return; }
      await STELLA.apiPost('/api/shipping/settings', { threshold, amount });
      STELLA.showToast('Parametres livraison sauvegardes');
      this.load();
    }
  },

  // ═══════════════════════════════════════════════════════════════
  // VUE 11: SYSTEME
  // ═══════════════════════════════════════════════════════════════
  system: {
    async load() {
      const [status, errors] = await Promise.all([
        STELLA.api('/api/system/status'),
        STELLA.api('/api/activity/log?limit=20&type=error')
      ]);
      if (!status) return;

      // Services
      const grid = document.getElementById('sys-services');
      if (grid && status.services) {
        grid.innerHTML = Object.entries(status.services).map(([name, s]) =>
          `<div class="service-item"><span class="status-dot ${s === 'online' ? 'online' : 'offline'}"></span><span class="service-name">${name}</span></div>`
        ).join('');
      }

      // Error badge
      const badge = document.getElementById('sys-errors-badge');
      if (badge) {
        badge.textContent = `${status.errors_24h || 0} erreurs 24h`;
        badge.className = `badge-pill badge-${(status.errors_24h || 0) > 0 ? 'error' : 'success'}`;
      }

      // Crons
      const cronFreq = { stock_check: '1h', sync_tags: 'Quotidien 3h15', nouveautes_expire: 'Quotidien 3h45', audit_qualite: 'Lundi 7h', tryme_expire: 'Quotidien 4h30', cashback_reminder: 'Quotidien 9h15', 'action-executor': '5 min', 'drift-detector': 'Quotidien 7h', 'daily-briefing': 'Quotidien 8h' };
      const cronTb = document.querySelector('#crons-table tbody');
      if (cronTb && status.crons) {
        cronTb.innerHTML = status.crons.map(c => `<tr>
          <td style="font-weight:600">${c.name || ''}</td>
          <td>${cronFreq[c.name] || '--'}</td>
          <td>${STELLA.relativeTime(c.executed_at)}</td>
          <td><span class="cron-status"><span class="cron-dot ${c.status === 'ok' ? 'ok' : 'error'}"></span> ${c.status || ''}</span></td>
        </tr>`).join('');
      }

      // Webhooks
      const whTb = document.querySelector('#webhooks-table tbody');
      if (whTb && status.webhooks) {
        whTb.innerHTML = status.webhooks.map(w => `<tr><td style="font-weight:600">${w.topic || ''}</td><td style="font-size:0.7rem;word-break:break-all">${w.url || ''}</td></tr>`).join('');
      }

      // Errors feed
      const errFeed = document.getElementById('sys-errors');
      if (errFeed && errors && errors.recent) {
        errFeed.innerHTML = errors.recent.slice(0, 20).map(e => `
          <div class="activity-item">
            <span class="activity-icon">🔴</span>
            <div class="activity-content">
              <div class="activity-action">${e.action || ''}</div>
              <div class="activity-detail">${e.source || ''}</div>
            </div>
            <span class="activity-time">${STELLA.relativeTime(e.timestamp)}</span>
          </div>`).join('') || '<div class="empty-state"><p>Aucune erreur</p></div>';
      }
    }
  },
  // ═══ STELLA-MEM — IA Metier ═══
  stellamem: {
    async load() {
      try {
        const [latest, reports, lessons] = await Promise.all([
          STELLA.api('/api/stella-mem/latest'),
          STELLA.api('/api/stella-mem/reports?limit=15'),
          STELLA.api('/api/stella-mem/lessons')
        ]);

        // KPIs from latest report
        const r = latest && latest.report ? latest.report : {};
        const fr = r.full_report || r;
        document.getElementById('sm-score').textContent = (fr.score !== undefined ? fr.score + '/10' : '-');
        document.getElementById('sm-sessions').textContent = fr.sessions_count || '-';
        document.getElementById('sm-actions').textContent = fr.total_actions || '-';
        const understood = fr.understood || r.understood || [];
        document.getElementById('sm-understood').textContent = Array.isArray(understood) ? understood.length : '-';

        // Lessons
        const lessonsEl = document.getElementById('sm-lessons');
        if (lessonsEl && lessons && lessons.suggestions) {
          lessonsEl.innerHTML = lessons.suggestions.map(s =>
            `<div class="activity-item"><span class="activity-icon">&#9888;</span><div class="activity-content"><div class="activity-action">${s}</div></div></div>`
          ).join('') || '<div class="empty-state"><p>Aucune lecon — le systeme apprend encore</p></div>';
        }

        // Problems
        const probsEl = document.querySelector('#sm-problems-table tbody');
        const problems = lessons && lessons.problems ? lessons.problems : (fr.problem_summary || {});
        if (probsEl) {
          const rows = Object.entries(problems).map(([type, data]) => {
            const d = typeof data === 'object' ? data : {count: data};
            return `<tr><td style="font-weight:600">${type}</td><td>${d.count || '-'}</td><td>${d.avg_severity || '-'}</td><td style="font-size:0.75rem">${(d.examples || []).join('; ').substring(0, 150)}</td></tr>`;
          });
          probsEl.innerHTML = rows.join('') || '<tr><td colspan="4" style="text-align:center">Aucun probleme recurrent</td></tr>';
        }

        // Suggestions
        const sugEl = document.getElementById('sm-suggestions');
        const sugs = fr.suggestions || [];
        if (sugEl) {
          sugEl.innerHTML = sugs.map(s =>
            `<div class="activity-item"><span class="activity-icon">&#128161;</span><div class="activity-content"><div class="activity-action">${s}</div></div></div>`
          ).join('') || '<div class="empty-state"><p>Pas encore de suggestions</p></div>';
        }

        // Not understood
        const nuEl = document.getElementById('sm-not-understood');
        const nu = fr.not_understood || r.not_understood || [];
        if (nuEl) {
          nuEl.innerHTML = (Array.isArray(nu) ? nu : []).map(item => {
            const desc = typeof item === 'object' ? `[${item.domain || item.type || '?'}] ${item.topic || item.description || ''}` : item;
            return `<div class="activity-item"><span class="activity-icon">&#10060;</span><div class="activity-content"><div class="activity-action">${desc}</div><div class="activity-detail">${item.reason || item.what_should_know || ''}</div></div></div>`;
          }).join('') || '<div class="empty-state"><p>Aucune lacune detectee</p></div>';
        }

        // Reports history
        const repTb = document.querySelector('#sm-reports-table tbody');
        if (repTb && reports && reports.reports) {
          repTb.innerHTML = reports.reports.map(rp => {
            const f = rp.full_report || rp;
            const u = rp.understood || f.understood || [];
            const n = rp.not_understood || f.not_understood || [];
            return `<tr>
              <td>${rp.report_date || rp.created_at?.substring(0, 10) || '-'}</td>
              <td>${rp.report_type || '-'}</td>
              <td style="font-weight:700;color:${rp.score >= 7 ? '#4CAF50' : rp.score >= 4 ? '#FF9800' : '#f44336'}">${rp.score !== null ? rp.score + '/10' : '-'}</td>
              <td>${rp.sessions_count || '-'}</td>
              <td>${rp.total_actions || '-'}</td>
              <td>${Array.isArray(u) ? u.length : '-'}</td>
              <td>${Array.isArray(n) ? n.length : '-'}</td>
            </tr>`;
          }).join('');
        }

      } catch (e) {
        console.error('STELLA-MEM load error:', e);
        document.getElementById('sm-lessons').innerHTML = '<div class="empty-state"><p>Erreur chargement STELLA-MEM</p></div>';
      }
    }
  },
  // ═══ MARKETING INTELLIGENCE ═══
  gads: {
    async load() {
      try {
        const r = await fetch('/api/marketing/intelligence');
        const d = await r.json();
        const $ = (id, v) => { const e = document.getElementById(id); if (e) e.textContent = v; };
        const $h = (id, v) => { const e = document.getElementById(id); if (e) e.innerHTML = v; };

        // Month label
        $('mi-month', 'Marketing Intelligence — ' + (d.month_label || ''));

        // ── Recommendations ──
        const recs = d.recommendations || [];
        if (recs.length) {
          document.getElementById('mi-recs-card').style.display = '';
          const colors = {critical:'#dc2626',high:'#f59e0b',medium:'#3b82f6'};
          $h('mi-recs', recs.map(r => `<div class="activity-item"><span style="color:${colors[r.priority]||'#888'}; font-weight:600">${r.priority.toUpperCase()}</span> <span style="margin-left:0.5rem">${r.text}</span></div>`).join(''));
        }

        // ── Budget gauge ──
        const b = d.budget || {};
        const ratio = b.ratio_pct || 0;
        const color = b.status === 'danger' ? 'var(--color-danger)' : b.status === 'warning' ? 'var(--color-warning)' : '#22c55e';
        const ratioEl = document.getElementById('mi-ratio');
        const bar = document.getElementById('mi-bar');
        if (ratioEl) { ratioEl.textContent = ratio.toFixed(2) + ' %'; ratioEl.style.color = color; }
        if (bar) { bar.style.width = Math.min(ratio / 10 * 100, 100) + '%'; bar.style.background = color; }
        $('mi-spend', STELLA.eur(b.ads_spend));
        $('mi-revenue', STELLA.eur(b.revenue));
        $('mi-proj-spend', STELLA.eur(b.projected_spend));
        const globalRoas = b.ads_spend > 0 ? (b.revenue / b.ads_spend).toFixed(1) + 'x' : '-';
        $('mi-roas', globalRoas);

        // ── Campaigns ──
        const camps = d.campaigns || [];
        $h('mi-campaigns', camps.filter(c => c.cost > 0 || c.status === 'ENABLED').map(c => {
          const roasColor = c.roas >= 8 ? '#22c55e' : c.roas >= 3 ? '#f59e0b' : c.roas > 0 ? '#dc2626' : '';
          const statusDot = c.status === 'ENABLED' ? '<span style="color:#22c55e">●</span>' : '<span style="color:#666">○</span>';
          return `<tr><td>${statusDot} ${c.name}</td><td style="font-size:0.7rem">${c.type.replace('PERFORMANCE_MAX','PMax').replace('DEMAND_GEN','DemGen')}</td><td>${STELLA.eur(c.cost)}</td><td>${c.clicks}</td><td>${c.conversions}</td><td>${STELLA.eur(c.conv_value)}</td><td style="font-weight:700;color:${roasColor}">${c.roas}x</td><td>${STELLA.eur(c.cpc)}</td><td>${c.ctr}%</td><td>${c.cpa ? STELLA.eur(c.cpa) : '-'}</td></tr>`;
        }).join('') || '<tr><td colspan="10" class="empty-state">Aucune campagne</td></tr>');

        // ── Asset Groups ──
        const ags = d.asset_groups || [];
        if (ags.length) {
          document.getElementById('mi-ag-card').style.display = '';
          $h('mi-asset-groups', ags.filter(a => a.cost > 0).map(a => {
            const rc = a.roas >= 8 ? '#22c55e' : a.roas >= 3 ? '#f59e0b' : '#dc2626';
            return `<tr><td>${a.name}</td><td>${STELLA.eur(a.cost)}</td><td>${a.clicks}</td><td>${a.conversions}</td><td>${STELLA.eur(a.conv_value)}</td><td style="font-weight:700;color:${rc}">${a.roas}x</td></tr>`;
          }).join(''));
        }

        // ── Funnel ──
        const f = d.funnel || {};
        $('mi-sessions', (f.sessions || 0).toLocaleString('fr-FR'));
        $('mi-atc-rate', (f.atc_rate || 0) + '%');
        $('mi-checkout-rate', (f.checkout_rate || 0) + '%');
        $('mi-purchase-rate', (f.purchase_rate || 0) + '%');
        $('mi-conv-rate', (f.overall_conv || 0) + '%');
        $('mi-ga4-revenue', STELLA.eur(f.revenue));

        // ── Channels ──
        $h('mi-channels', (d.channels || []).map(ch => {
          const bounceColor = ch.bounce_rate > 80 ? 'color:#dc2626' : ch.bounce_rate > 50 ? 'color:#f59e0b' : '';
          return `<tr><td style="font-weight:500">${ch.name}</td><td>${ch.sessions}</td><td>${ch.atc}</td><td>${ch.checkout}</td><td>${ch.purchases}</td><td>${STELLA.eur(ch.revenue)}</td><td style="${bounceColor}">${ch.bounce_rate}%</td><td>${ch.conv_rate}%</td></tr>`;
        }).join('') || '<tr><td colspan="8" class="empty-state">Aucune donnee</td></tr>');

        // ── Conversion Tracking ──
        const ct = d.conversion_tracking || {};
        const primary = ct.primary_actions || [];
        const badge = document.getElementById('mi-tracking-badge');
        if (badge) {
          badge.textContent = ct.purchase_tracking ? 'OK' : 'ALERTE';
          badge.style.background = ct.purchase_tracking ? '#22c55e' : '#dc2626';
        }
        let trackHtml = '<div style="margin-bottom:0.5rem"><strong>Actions primaires (' + primary.length + ') :</strong></div>';
        trackHtml += primary.map(a => `<div class="activity-item"><span style="color:#22c55e">●</span> ${a.name} <span style="color:var(--text-muted);font-size:0.7rem">${a.category} / ${a.type}</span></div>`).join('');
        if (ct.secondary_actions && ct.secondary_actions.length) {
          trackHtml += '<div style="margin-top:0.75rem;margin-bottom:0.5rem"><strong>Secondaires (' + ct.secondary_actions.length + ') :</strong></div>';
          trackHtml += ct.secondary_actions.slice(0, 5).map(a => `<div class="activity-item"><span style="color:#666">○</span> ${a.name} <span style="color:var(--text-muted);font-size:0.7rem">${a.category}</span></div>`).join('');
          if (ct.secondary_actions.length > 5) trackHtml += `<div style="color:var(--text-muted);font-size:0.75rem">+ ${ct.secondary_actions.length - 5} autres</div>`;
        }
        $h('mi-tracking', trackHtml);

        // ── SEO ──
        const seo = d.seo || {};
        $h('mi-seo-queries', (seo.top_queries || []).map(q =>
          `<tr><td>${q.query}</td><td>${q.clicks}</td><td>${q.impressions}</td><td>${q.ctr}%</td><td>${q.position}</td></tr>`
        ).join('') || '<tr><td colspan="5" class="empty-state">-</td></tr>');
        $h('mi-seo-pages', (seo.top_pages || []).map(p =>
          `<tr><td style="font-size:0.75rem">${p.page || '/'}</td><td>${p.clicks}</td><td>${p.impressions}</td><td>${p.ctr}%</td><td>${p.position}</td></tr>`
        ).join('') || '<tr><td colspan="5" class="empty-state">-</td></tr>');

        // ── Merchant Center ──
        const mc = d.merchant || {};
        $h('mi-merchant-kpis', `
          <div class="kpi-card"><div class="kpi-value">${mc.total_products || 0}</div><div class="kpi-label">Produits</div></div>
          <div class="kpi-card"><div class="kpi-value" style="color:#22c55e">${mc.ok || 0}</div><div class="kpi-label">OK</div></div>
          <div class="kpi-card"><div class="kpi-value" style="color:#f59e0b">${mc.warnings || 0}</div><div class="kpi-label">Warnings</div></div>
          <div class="kpi-card"><div class="kpi-value" style="color:#dc2626">${mc.disapproved || 0}</div><div class="kpi-label">Disapproved</div></div>
        `);
        if (mc.top_issues && mc.top_issues.length) {
          $h('mi-merchant-issues', '<div style="font-size:0.75rem;color:var(--text-muted)">Top issues: ' +
            mc.top_issues.slice(0, 5).map(i => `<span style="margin-right:0.75rem">${i.code} (${i.count})</span>`).join('') + '</div>');
        }

        // ── Daily breakdown ──
        $h('mi-daily', (b.daily || []).map(row => {
          const r2 = row.revenue > 0 ? (row.spend / row.revenue * 100).toFixed(1) + '%' : '-';
          return `<tr><td>${row.date}</td><td>${STELLA.eur(row.spend)}</td><td>${STELLA.eur(row.revenue)}</td><td>${r2}</td></tr>`;
        }).join('') || '<tr><td colspan="4" class="empty-state">-</td></tr>');

        // Errors
        if (d.errors && d.errors.length) console.warn('Marketing Intel errors:', d.errors);
      } catch (e) {
        console.error('Marketing Intelligence load error:', e);
      }
    }
  }
};

// Boot
document.addEventListener('DOMContentLoaded', () => STELLA.init());
