(() => {
    const notices = document.getElementById('ui-notices');
    const dialog = document.getElementById('confirm-dialog');

    /**
     * Mostra per cinque secondi una notifica nell'area degli avvisi.
     * @param {string} message Testo della notifica.
     * @param {string} [type='info'] Tipo di notifica (`info`, `success`, `error`), usato come classe CSS.
     */
    function notify(message, type = 'info') {
        if (!notices || !message) return;
        const notice = document.createElement('div');
        notice.className = `notice notice--${type}`;
        notice.setAttribute('role', type === 'error' ? 'alert' : 'status');
        notice.textContent = message;
        notices.appendChild(notice);
        window.setTimeout(() => notice.remove(), 5000);
    }

    /**
     * Attiva o disattiva lo stato di attesa di un pulsante.
     * @param {HTMLButtonElement} button Pulsante da aggiornare.
     * @param {boolean} busy True per disabilitarlo, false per ripristinarlo.
     * @param {string} [label] Etichetta da mostrare; se manca, al ripristino torna quella precedente.
     */
    function setBusy(button, busy, label) {
        if (!button) return;
        if (busy) {
            button.dataset.label = button.textContent;
            button.disabled = true;
            button.classList.add('is-busy');
            if (label) button.textContent = label;
            return;
        }
        button.disabled = false;
        button.classList.remove('is-busy');
        button.textContent = label || button.dataset.label || button.textContent;
    }

    /**
     * Chiede conferma all'utente con la finestra di dialogo della pagina, o con `window.confirm` dove manca.
     * @param {object} [options] Testi della finestra.
     * @param {string} [options.title='Conferma azione'] Titolo.
     * @param {string} [options.message=''] Messaggio.
     * @param {string} [options.confirmLabel='Conferma'] Etichetta del pulsante di conferma.
     * @returns {Promise<boolean>} True se l'utente conferma.
     */
    function confirmAction({ title = 'Conferma azione', message = '', confirmLabel = 'Conferma' } = {}) {
        if (!dialog || typeof dialog.showModal !== 'function') {
            return Promise.resolve(window.confirm(message));
        }
        document.getElementById('confirm-title').textContent = title;
        document.getElementById('confirm-message').textContent = message;
        dialog.querySelector('button[value="confirm"]').textContent = confirmLabel;
        dialog.showModal();
        return new Promise((resolve) => {
            dialog.addEventListener('close', () => resolve(dialog.returnValue === 'confirm'), { once: true });
        });
    }

    document.addEventListener('keydown', (event) => {
        if ((event.ctrlKey || event.metaKey) && event.key === 'Enter') {
            const form = event.target.closest('form');
            const submit = form?.querySelector('button[type="submit"]:not(:disabled)');
            if (submit) {
                event.preventDefault();
                submit.click();
            }
        }
    });

    window.UI = { notify, setBusy, confirmAction };
})();
