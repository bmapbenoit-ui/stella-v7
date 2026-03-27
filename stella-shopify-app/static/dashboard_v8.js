/* ═══════════════════════════════════════════════════════════════
   STELLA V8 — Dashboard JavaScript
   Tour de contrôle PlanèteBeauty
   ═══════════════════════════════════════════════════════════════ */

const STELLA = {
  state: { activeTab: 'home', loaded: {}, cache: {} },

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
    document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    const panel = document.getElementById(`panel-${tabId}`);
    const btn = document.querySelector(`[data-tab="${tabId}"]`);
    if (panel) panel.classList.add('active');
    if (btn) btn.classList.add('active');
    this.state.activeTab = tabId;
    if (!this.state.loaded[tabId]) {
      if (this[tabId] && this[tabId].load) this[tabId].load();
      this.state.loaded[tabId] = Date.now();
    } else if (Date.now() - this.state.loaded[tabId] > 300000) {
      if (this[tabId] && this[tabId].load) this[tabId].load();
      this.state.loaded[tabId] = Date.now();
    }
  },

  refreshAll() {
    this.state.loaded = {};
    if (this[this.state.activeTab] && this[this.state.activeTab].load) {
      this[this.state.activeTab].load();
    }
    this.showToast('Rafraichi', 'success');
  },

  // ═══ API ═══
  async api(url) {
    try {
      const r = await fetch(url);
      if (!r.ok) throw new Error(`${r.status}`);
      return await r.json();
    } catch (e) {
      console.error(`API ${url}:`, e);
      return null;
    }
  },

  async apiPost(url, body) {
    try {
      const r = await fetch(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
      return await r.json();
    } catch (e) {
      console.error(`API POST ${url}:`, e);
      return null;
    }
  },

  // ═══ UTILS ═══
  eur(n) { return (n || 0).toLocaleString('fr-FR', { minimumFractionDigits: 2, maximumFractionDigits: 2 }) + ' \u20ac'; },
  shortTime(iso) {
    if (!iso) return '--';
    const d = new Date(iso);
    return d.toLocaleString('fr-FR', { day: '2-digit', month: '2-digit', hour: '2-digit', minute: '2-digit', timeZone: 'Europe/Paris' });
  },
  relativeTime(iso) {
    if (!iso) return '--';
    const diff = (Date.now() - new Date(iso).getTime()) / 1000;
    if (diff < 60) return 'a l\'instant';
    if (diff < 3600) return Math.floor(diff / 60) + ' min';
    if (diff < 86400) return Math.floor(diff / 3600) + ' h';
    return Math.floor(diff / 86400) + ' j';
  },
  statusBadge(status) {
    const map = { PAID: 'success', PARTIALLY_PAID: 'success', UNFULFILLED: 'warning', FULFILLED: 'success', success: 'success', error: 'error', active: 'info', notified: 'success' };
    return `<span class="badge-pill badge-${map[status] || 'info'}">${status}</span>`;
  },
  activityIcon(type) {
    const map = { cashback_credit: '\u{1F4B0}', cron_run: '\u23F0', product_change: '\u{1F4E6}', bis_notification: '\u{1F514}', email_sent: '\u2709\uFE0F', error: '\u26A0\uFE0F', webhook_received: '\u{1F517}', trustpilot_credit: '\u2B50' };
    return map[type] || '\u26AA';
  },

  showToast(msg, type = 'success') {
    const t = document.createElement('div');
    t.className = `toast ${type}`;
    t.textContent = msg;
    document.body.appendChild(t);
    setTimeout(() => t.remove(), 3000);
  },

  // ═══════════════════════════════════════════════════════════════
  // TAB 1: ACCUEIL
  // ═══════════════════════════════════════════════════════════════
  home: {
    async load() {
      const [kpis, activity, system] = await Promise.all([
        STELLA.api('/api/kpis/summary'),
        STELLA.api('/api/activity/log?limit=20'),
        STELLA.api('/api/system/status')
      ]);

      if (kpis) {
        document.getElementById('kpi-revenue').textContent = STELLA.eur(kpis.revenue_today);
        document.getElementById('kpi-orders').textContent = kpis.orders_today || 0;
        document.getElementById('kpi-aov').textContent = STELLA.eur(kpis.avg_order_value);
        document.getElementById('kpi-cashback').textContent = STELLA.eur(kpis.cashback_generated_today);
        document.getElementById('badge-orders').textContent = kpis.orders_today || 0;
        document.getElementById('badge-bis').textContent = kpis.bis_active || 0;
      }

      if (activity && activity.activities) {
        const feed = document.getElementById('activity-feed');
        document.getElementById('activity-count').textContent = activity.total || 0;
        if (activity.activities.length === 0) {
          feed.innerHTML = '<div class="empty-state"><div class="icon">\u{1F4CB}</div><p>Aucune activite enregistree</p></div>';
        } else {
          feed.innerHTML = activity.activities.map(a => `
            <div class="activity-item">
              <span class="activity-icon">${STELLA.activityIcon(a.type)}</span>
              <div class="activity-content">
                <div class="activity-action">${a.action}</div>
                <div class="activity-detail">${a.customer_email || a.order_name || a.product_title || ''} ${a.status === 'error' ? '<span class="badge-pill badge-error">ERREUR</span>' : ''}</div>
              </div>
              <span class="activity-time">${STELLA.relativeTime(a.timestamp)}</span>
            </div>
          `).join('');
        }
      }

      if (system && system.services) {
        const grid = document.getElementById('services-grid');
        const dot = document.getElementById('global-status');
        const allOnline = Object.values(system.services).every(s => s === 'online');
        dot.className = `status-dot ${allOnline ? 'online' : 'offline'}`;
        grid.innerHTML = Object.entries(system.services).map(([name, status]) =>
          `<div class="service-item"><span class="status-dot ${status === 'online' ? 'online' : 'offline'}"></span><span class="service-name">${name}</span></div>`
        ).join('');
      }
    }
  },

  // ═══════════════════════════════════════════════════════════════
  // TAB 2: COMMANDES
  // ═══════════════════════════════════════════════════════════════
  orders: {
    async load() {
      const data = await STELLA.api('/api/orders/dashboard');
      if (!data) return;
      const kpis = document.getElementById('orders-kpis');
      const paid = data.orders.filter(o => ['PAID', 'PARTIALLY_PAID'].includes(o.financial_status));
      kpis.innerHTML = `
        <div class="kpi-card"><div class="kpi-value gold">${STELLA.eur(data.revenue_today)}</div><div class="kpi-label">CA Jour</div></div>
        <div class="kpi-card"><div class="kpi-value">${data.count}</div><div class="kpi-label">Commandes</div></div>
        <div class="kpi-card"><div class="kpi-value">${paid.length}</div><div class="kpi-label">Payees</div></div>
        <div class="kpi-card"><div class="kpi-value">${data.orders.filter(o => o.fulfillment_status === 'UNFULFILLED').length}</div><div class="kpi-label">A expedier</div></div>
      `;
      const tbody = document.querySelector('#orders-table tbody');
      if (data.orders.length === 0) {
        tbody.innerHTML = '<tr><td colspan="5" style="text-align:center;color:var(--text-muted)">Aucune commande aujourd\'hui</td></tr>';
      } else {
        tbody.innerHTML = data.orders.map(o => `
          <tr>
            <td><strong>${o.name}</strong></td>
            <td>${o.customer_name || o.customer_email || '--'}</td>
            <td>${STELLA.eur(o.total)}</td>
            <td>${STELLA.statusBadge(o.financial_status)} ${STELLA.statusBadge(o.fulfillment_status)}</td>
            <td>${(o.tags || []).map(t => `<span class="badge-pill badge-gold">${t}</span>`).join(' ')}</td>
          </tr>
        `).join('');
      }
    }
  },

  // ═══════════════════════════════════════════════════════════════
  // TAB 3: CASHBACK
  // ═══════════════════════════════════════════════════════════════
  cashback: {
    async load() {
      const [dashboard, settings] = await Promise.all([
        STELLA.api('/api/cashback/dashboard'),
        STELLA.api('/api/cashback/settings')
      ]);
      if (dashboard) {
        const pendingCount = (dashboard.pending || []).length;
        const pendingAmount = (dashboard.pending || []).reduce((s, r) => s + parseFloat(r.cashback_amount || 0), 0);
        document.getElementById('cb-kpis').innerHTML = `
          <div class="kpi-card"><div class="kpi-value gold">${dashboard.total_rewarded || 0}</div><div class="kpi-label">Credits generes</div></div>
          <div class="kpi-card"><div class="kpi-value">${STELLA.eur(dashboard.total_amount || 0)}</div><div class="kpi-label">Montant total</div></div>
          <div class="kpi-card"><div class="kpi-value">${pendingCount}</div><div class="kpi-label">Encours actifs</div></div>
          <div class="kpi-card"><div class="kpi-value">${STELLA.eur(pendingAmount)}</div><div class="kpi-label">Encours EUR</div></div>
        `;

        // Expiring soon table
        const expiring = dashboard.expiring_soon || [];
        document.getElementById('cb-expiring-count').textContent = expiring.length;
        const expTbody = document.querySelector('#cb-expiring-table tbody');
        expTbody.innerHTML = expiring.length ? expiring.map(r => `
          <tr>
            <td>${r.customer_email || '--'}</td>
            <td>${r.order_name || '--'}</td>
            <td style="font-weight:600;color:var(--gold)">${STELLA.eur(r.cashback_amount)}</td>
            <td style="color:var(--error)">${STELLA.shortTime(r.expires_at)}</td>
            <td>${r.reminder_sent ? '<span style="color:var(--success)">Envoye</span>' : '<span style="color:var(--text-muted)">En attente</span>'}</td>
          </tr>
        `).join('') : '<tr><td colspan="5" style="text-align:center;color:var(--text-muted)">Aucun cashback a expiration proche</td></tr>';

        // Pending table
        const pending = dashboard.pending || [];
        document.getElementById('cb-pending-count').textContent = pendingCount;
        const pendTbody = document.querySelector('#cb-pending-table tbody');
        pendTbody.innerHTML = pending.length ? pending.map(r => `
          <tr>
            <td>${r.customer_email || '--'}</td>
            <td>${r.order_name || '--'}</td>
            <td>${STELLA.eur(r.cashback_amount)}</td>
            <td>${STELLA.shortTime(r.expires_at)}</td>
          </tr>
        `).join('') : '<tr><td colspan="4" style="text-align:center;color:var(--text-muted)">Aucun encours</td></tr>';

        // Flow table (recent with full details)
        const recent = dashboard.recent || [];
        const flowTbody = document.querySelector('#cb-table tbody');
        flowTbody.innerHTML = recent.length ? recent.map(r => {
          const statusBadge = r.status === 'active' ? '<span style="color:var(--success)">Actif</span>'
            : r.status === 'revoked' ? '<span style="color:var(--error)">Annule</span>'
            : r.status === 'used' ? '<span style="color:var(--text-muted)">Utilise</span>'
            : r.status || '--';
          const emailIcon = r.email_sent ? '\u2709\uFE0F' : '\u274C';
          const reminderInfo = r.reminder_sent_at ? ' \u23F0' : '';
          return `<tr>
            <td>${r.customer_email || '--'}</td>
            <td>${r.order_name || '--'}</td>
            <td>${STELLA.eur(r.cashback_base)}</td>
            <td style="font-weight:600">${STELLA.eur(r.cashback_amount)}</td>
            <td>${statusBadge}</td>
            <td>${emailIcon}${reminderInfo}</td>
            <td>${STELLA.shortTime(r.created_at)}</td>
          </tr>`;
        }).join('') : '<tr><td colspan="7" style="text-align:center;color:var(--text-muted)">Aucun cashback</td></tr>';
      }
      if (settings) {
        document.getElementById('cb-rate').value = (settings.cashback_rate || 0.05) * 100;
        document.getElementById('cb-expiry').value = settings.expiry_days || 60;
        document.getElementById('cb-min-use').value = settings.min_order_use || 70;
        document.getElementById('cb-min-amount').value = settings.min_cashback_amount || 0.5;
        document.getElementById('cb-excluded-tags').value = settings.excluded_tags || '';
      }
    },
    async save() {
      const body = {
        cashback_rate: parseFloat(document.getElementById('cb-rate').value) / 100,
        expiry_days: parseInt(document.getElementById('cb-expiry').value),
        min_order_use: parseFloat(document.getElementById('cb-min-use').value),
        min_cashback_amount: parseFloat(document.getElementById('cb-min-amount').value),
        excluded_tags: document.getElementById('cb-excluded-tags').value
      };
      const r = await STELLA.apiPost('/api/cashback/settings', body);
      STELLA.showToast(r && r.ok ? 'Cashback sauvegarde' : 'Erreur sauvegarde', r && r.ok ? 'success' : 'error');
    }
  },

  // ═══════════════════════════════════════════════════════════════
  // TAB 5: QUIZ
  // ═══════════════════════════════════════════════════════════════
  quiz: {
    async load() {
      const data = await STELLA.api('/api/quiz/stats');
      if (!data) return;
      document.getElementById('quiz-kpis').innerHTML = `
        <div class="kpi-card"><div class="kpi-value gold">${data.today?.quiz_view || 0}</div><div class="kpi-label">Vues aujourd'hui</div></div>
        <div class="kpi-card"><div class="kpi-value">${data.today?.quiz_complete || 0}</div><div class="kpi-label">Completions</div></div>
        <div class="kpi-card"><div class="kpi-value">${data.today?.quiz_add_to_cart || 0}</div><div class="kpi-label">Ajouts panier</div></div>
        <div class="kpi-card"><div class="kpi-value">${data.products_count || 0}</div><div class="kpi-label">Produits indexes</div></div>
      `;
      const details = document.getElementById('quiz-details');
      details.innerHTML = `
        <p style="color:var(--text-secondary);font-size:0.85rem">
          Derniere regeneration : <strong>${data.last_regenerated || '--'}</strong><br>
          Page : <a href="https://planetebeauty.com/pages/quiz-olfactif" target="_blank" style="color:var(--gold)">planetebeauty.com/pages/quiz-olfactif</a>
        </p>
      `;
    },
    async regenerate() {
      STELLA.showToast('Regeneration en cours...', 'success');
      const r = await STELLA.apiPost('/api/quiz/regenerate', {});
      STELLA.showToast(r && r.success ? `Quiz regenere: ${r.count} produits` : 'Erreur', r && r.success ? 'success' : 'error');
      this.load();
    }
  },

  // ═══════════════════════════════════════════════════════════════
  // TAB 6: BACK IN STOCK
  // ═══════════════════════════════════════════════════════════════
  bis: {
    async load() {
      const [data, smtp] = await Promise.all([
        STELLA.api('/api/bis/dashboard'),
        STELLA.api('/api/bis/smtp-status')
      ]);
      if (data) {
        document.getElementById('bis-kpis').innerHTML = `
          <div class="kpi-card"><div class="kpi-value gold">${data.total_active || 0}</div><div class="kpi-label">Abonnes actifs</div></div>
          <div class="kpi-card"><div class="kpi-value">${data.total_products || 0}</div><div class="kpi-label">Produits surveilles</div></div>
          <div class="kpi-card"><div class="kpi-value">${data.total_notified || 0}</div><div class="kpi-label">Emails envoyes</div></div>
          <div class="kpi-card"><div class="kpi-value">${data.total_subscriptions || 0}</div><div class="kpi-label">Total inscriptions</div></div>
        `;
        const tbody = document.querySelector('#bis-table tbody');
        const subs = data.recent || data.subscriptions || [];
        tbody.innerHTML = subs.length ? subs.slice(0, 20).map(s => `
          <tr><td>${s.email}</td><td>${s.product_title || '--'}</td><td>${STELLA.shortTime(s.subscribed_at)}</td><td>${STELLA.statusBadge(s.status)}</td></tr>
        `).join('') : '<tr><td colspan="4" style="text-align:center;color:var(--text-muted)">Aucun abonne</td></tr>';
      }
      if (smtp) {
        const badge = document.getElementById('bis-smtp-status');
        badge.textContent = `SMTP: ${smtp.status || 'unknown'}`;
        badge.className = `badge-pill badge-${smtp.status === 'ok' ? 'success' : 'error'}`;
      }
    }
  },

  // ═══════════════════════════════════════════════════════════════
  // TAB 7: AVIS & TRUSTPILOT
  // ═══════════════════════════════════════════════════════════════
  reviews: {
    async load() {
      const [reviews, trustpilot] = await Promise.all([
        STELLA.api('/api/reviews/dashboard'),
        STELLA.api('/api/trustpilot/dashboard')
      ]);
      if (reviews) {
        document.getElementById('reviews-kpis').innerHTML = `
          <div class="kpi-card"><div class="kpi-value gold">${reviews.total || 0}</div><div class="kpi-label">Total avis</div></div>
          <div class="kpi-card"><div class="kpi-value">${reviews.avg_rating || 0}/5</div><div class="kpi-label">Note moyenne</div></div>
          <div class="kpi-card"><div class="kpi-value">${trustpilot ? trustpilot.total_credits : 0}</div><div class="kpi-label">Credits Trustpilot</div></div>
          <div class="kpi-card"><div class="kpi-value">${trustpilot ? STELLA.eur(trustpilot.total_amount) : '--'}</div><div class="kpi-label">Montant credite</div></div>
        `;
        const tbody = document.querySelector('#reviews-table tbody');
        const recent = reviews.recent || [];
        tbody.innerHTML = recent.length ? recent.map(r => `
          <tr><td>${r.product_handle || r.product_title || '--'}</td><td>${'\u2B50'.repeat(Math.round(r.rating || 0))}</td><td>${r.source || '--'}</td><td>${STELLA.shortTime(r.created_at)}</td></tr>
        `).join('') : '<tr><td colspan="4" style="text-align:center;color:var(--text-muted)">Aucun avis</td></tr>';
      }
      if (trustpilot) {
        const list = document.getElementById('trustpilot-list');
        const recent = trustpilot.recent || [];
        if (recent.length === 0) {
          list.innerHTML = '<div class="empty-state"><p>Aucun credit Trustpilot</p></div>';
        } else {
          list.innerHTML = recent.map(t => `
            <div class="activity-item">
              <span class="activity-icon">\u2B50</span>
              <div class="activity-content"><div class="activity-action">${t.customer_email || '--'} — ${STELLA.eur(t.amount)}</div></div>
              <span class="activity-time">${STELLA.shortTime(t.credited_at)}</span>
            </div>
          `).join('');
        }
      }
    },
    async forceScan() {
      STELLA.showToast('Scan Trustpilot en cours...', 'success');
      const r = await STELLA.apiPost('/api/cron/trustpilot-scan', {});
      STELLA.showToast(r ? 'Scan termine' : 'Erreur', r ? 'success' : 'error');
      this.load();
    }
  },

  // ═══════════════════════════════════════════════════════════════
  // TAB 8: CATALOGUE
  // ═══════════════════════════════════════════════════════════════
  catalogue: {
    async load() {
      const data = await STELLA.api('/api/catalogue/dashboard');
      if (!data) return;
      document.getElementById('catalogue-kpis').innerHTML = `
        <div class="kpi-card"><div class="kpi-value gold">${data.total_products || 0}</div><div class="kpi-label">Total produits</div></div>
        <div class="kpi-card"><div class="kpi-value">${data.products_with_issues || 0}</div><div class="kpi-label">Avec problemes</div></div>
        <div class="kpi-card"><div class="kpi-value">${data.last_audit ? STELLA.relativeTime(data.last_audit) : '--'}</div><div class="kpi-label">Dernier audit</div></div>
      `;
      const tbody = document.querySelector('#catalogue-issues tbody');
      const issues = data.issues || [];
      tbody.innerHTML = issues.length ? issues.map(i => `
        <tr><td>${i.title || i.product || '--'}</td><td>${(i.problems || i.issues || []).join(', ')}</td></tr>
      `).join('') : '<tr><td colspan="2" style="text-align:center;color:var(--text-muted)">Aucun probleme detecte</td></tr>';
    },
    async forceAudit() {
      STELLA.showToast('Audit en cours...', 'success');
      await STELLA.apiPost('/api/cron/audit-qualite', {});
      STELLA.showToast('Audit lance', 'success');
    },
    async syncTags() {
      STELLA.showToast('Sync tags en cours...', 'success');
      await STELLA.apiPost('/api/cron/sync-tags', {});
      STELLA.showToast('Sync lancee', 'success');
    }
  },

  // ═══════════════════════════════════════════════════════════════
  // TAB 9: SYSTEME
  // ═══════════════════════════════════════════════════════════════
  system: {
    async load() {
      const [status, errors] = await Promise.all([
        STELLA.api('/api/system/status'),
        STELLA.api('/api/activity/log?limit=20&type=error')
      ]);

      if (status) {
        // Services
        const grid = document.getElementById('sys-services');
        grid.innerHTML = Object.entries(status.services || {}).map(([name, s]) =>
          `<div class="service-item"><span class="status-dot ${s === 'online' ? 'online' : 'offline'}"></span><span class="service-name">${name}</span></div>`
        ).join('');

        // Errors badge
        const badge = document.getElementById('sys-errors-badge');
        badge.textContent = `${status.errors_24h || 0} erreurs 24h`;
        badge.className = `badge-pill badge-${status.errors_24h > 0 ? 'error' : 'success'}`;

        // Crons
        const cronBody = document.querySelector('#crons-table tbody');
        const crons = status.crons || [];
        cronBody.innerHTML = crons.length ? crons.map(c => `
          <tr>
            <td><strong>${c.cron_name}</strong></td>
            <td>${STELLA.shortTime(c.executed_at)}</td>
            <td><span class="cron-status"><span class="cron-dot ${c.status === 'error' ? 'error' : 'ok'}"></span> ${c.status}</span></td>
          </tr>
        `).join('') : '<tr><td colspan="3" style="text-align:center;color:var(--text-muted)">Aucun cron enregistre</td></tr>';

        // Webhooks
        const whBody = document.querySelector('#webhooks-table tbody');
        const wh = status.webhooks || [];
        whBody.innerHTML = wh.length ? wh.map(w => `
          <tr><td>${w.topic}</td><td style="font-size:0.7rem;color:var(--text-muted);word-break:break-all">${w.url}</td></tr>
        `).join('') : '<tr><td colspan="2" style="text-align:center;color:var(--text-muted)">Aucun webhook</td></tr>';
      }

      // Errors feed
      if (errors && errors.activities) {
        const el = document.getElementById('sys-errors');
        if (errors.activities.length === 0) {
          el.innerHTML = '<div class="empty-state"><div class="icon">\u2705</div><p>Aucune erreur recente</p></div>';
        } else {
          el.innerHTML = errors.activities.map(a => `
            <div class="activity-item">
              <span class="activity-icon">\u26A0\uFE0F</span>
              <div class="activity-content">
                <div class="activity-action">${a.action}</div>
                <div class="activity-detail" style="color:var(--error)">${a.type} — ${a.source}</div>
              </div>
              <span class="activity-time">${STELLA.relativeTime(a.timestamp)}</span>
            </div>
          `).join('');
        }
      }
    }
  },

  // ═══════════════════════════════════════════════════════════════
  // TAB 10: CODES PROMO
  // ═══════════════════════════════════════════════════════════════
  promo: {
    _config: null,
    async load() {
      const data = await STELLA.api('/api/promo/settings');
      if (data) {
        this._config = data;
        this.render();
      }
    },
    render() {
      const codes = this._config?.codes || {};
      const vendors = this._config?.excludedVendors || [];
      const tbody = document.getElementById('promo-tbody');
      const entries = Object.entries(codes);
      tbody.innerHTML = entries.length ? entries.map(([name, r]) => `
        <tr>
          <td><strong style="color:var(--gold)">${name}</strong></td>
          <td>-${r.percent}%</td>
          <td>${r.minSubtotal}\u20AC+</td>
          <td style="font-size:0.8rem">${r.message || ''}</td>
          <td>
            <button onclick="STELLA.promo.editCode('${name}')" style="background:none;border:none;cursor:pointer;font-size:0.85rem">\u270F\uFE0F</button>
            <button onclick="STELLA.promo.deleteCode('${name}')" style="background:none;border:none;cursor:pointer;font-size:0.85rem">\u274C</button>
          </td>
        </tr>
      `).join('') : '<tr><td colspan="5" style="text-align:center;color:var(--text-muted)">Aucun code</td></tr>';
      document.getElementById('promo-excluded-vendors').value = vendors.join(', ');
    },
    addCode() {
      const name = prompt('Nom du code (ex: PB580):');
      if (!name) return;
      const upper = name.toUpperCase().trim();
      const percent = parseFloat(prompt('Pourcentage de reduction (ex: 5):'));
      const minSubtotal = parseFloat(prompt('Montant minimum du panier (ex: 80):'));
      if (!percent || !minSubtotal) return STELLA.showToast('Valeurs invalides', 'error');
      const message = `-${percent}% avec le code ${upper}`;
      if (!this._config) this._config = {codes: {}, excludedVendors: []};
      this._config.codes[upper] = {percent, minSubtotal, message};
      this.render();
      STELLA.showToast(`Code ${upper} ajoute (non sauvegarde)`, 'success');
    },
    editCode(name) {
      const rule = this._config?.codes?.[name];
      if (!rule) return;
      const percent = parseFloat(prompt(`Nouveau % pour ${name} (actuel: ${rule.percent}):`, rule.percent));
      const minSubtotal = parseFloat(prompt(`Nouveau min panier pour ${name} (actuel: ${rule.minSubtotal}):`, rule.minSubtotal));
      if (!percent || !minSubtotal) return;
      rule.percent = percent;
      rule.minSubtotal = minSubtotal;
      rule.message = `-${percent}% avec le code ${name}`;
      this.render();
      STELLA.showToast(`Code ${name} modifie (non sauvegarde)`, 'success');
    },
    deleteCode(name) {
      if (!confirm(`Supprimer le code ${name} ?`)) return;
      delete this._config.codes[name];
      this.render();
      STELLA.showToast(`Code ${name} supprime (non sauvegarde)`, 'success');
    },
    async save() {
      const vendorsRaw = document.getElementById('promo-excluded-vendors').value;
      const excludedVendors = vendorsRaw.split(',').map(v => v.trim()).filter(Boolean);
      this._config.excludedVendors = excludedVendors;
      STELLA.showToast('Sauvegarde en cours...', 'success');
      const r = await STELLA.apiPost('/api/promo/settings', this._config);
      if (r && r.success) {
        STELLA.showToast(`${r.codes_count} codes promo sauvegardes`, 'success');
      } else {
        STELLA.showToast('Erreur sauvegarde', 'error');
      }
    }
  },

  // ═══════════════════════════════════════════════════════════════
  // TAB 11: LIVRAISON
  // ═══════════════════════════════════════════════════════════════
  shipping: {
    async load() {
      const data = await STELLA.api('/api/shipping/settings');
      if (data) {
        document.getElementById('shipping-threshold').value = data.threshold || 99;
        document.getElementById('shipping-amount').value = data.amount || 5;
        document.getElementById('shipping-status').innerHTML = `
          <span class="badge-pill badge-success">ACTIF</span>
          <span style="margin-left:8px;color:var(--text-secondary)">-${data.amount || 5}€ dès ${data.threshold || 99}€</span>
        `;
      }
    },
    async save() {
      const threshold = parseFloat(document.getElementById('shipping-threshold').value);
      const amount = parseFloat(document.getElementById('shipping-amount').value);
      if (!threshold || !amount || threshold <= 0 || amount <= 0) {
        return STELLA.showToast('Valeurs invalides', 'error');
      }
      STELLA.showToast('Sauvegarde en cours...', 'success');
      const r = await STELLA.apiPost('/api/shipping/settings', { threshold, amount });
      if (r && r.success) {
        STELLA.showToast(`Livraison mis à jour: -${amount}€ dès ${threshold}€`, 'success');
        this.load();
      } else {
        STELLA.showToast('Erreur sauvegarde', 'error');
      }
    }
  }
};

// ═══ BOOT ═══
document.addEventListener('DOMContentLoaded', () => STELLA.init());
