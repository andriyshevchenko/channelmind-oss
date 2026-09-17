import { ref, computed, onMounted, onBeforeUnmount } from 'vue';

// Custom dropdown replacing native <select> (native option lists can't be styled).
// Keyboard: Enter/Space/ArrowDown opens, arrows move, Enter picks, Escape closes.
export default {
  props: {
    modelValue: {},
    options: { type: Array, required: true }, // strings or {value, label}
    small: Boolean,
    disabled: Boolean,
  },
  emits: ['update:modelValue'],
  setup(props, { emit }) {
    const open = ref(false);
    const hi = ref(-1);
    const root = ref(null);
    const norm = computed(() => props.options.map(o => (o && typeof o === 'object') ? o : { value: o, label: String(o) }));
    const label = computed(() => (norm.value.find(o => o.value === props.modelValue) || {}).label ?? String(props.modelValue ?? ''));
    const selIndex = () => norm.value.findIndex(o => o.value === props.modelValue);
    const pick = (v) => { emit('update:modelValue', v); open.value = false; };
    const toggle = () => { if (props.disabled) return; open.value = !open.value; if (open.value) hi.value = Math.max(0, selIndex()); };
    const onKey = (e) => {
      if (!open.value) {
        if (['Enter', ' ', 'ArrowDown', 'ArrowUp'].includes(e.key)) { e.preventDefault(); open.value = true; hi.value = Math.max(0, selIndex()); }
        return;
      }
      if (e.key === 'Escape') { open.value = false; }
      else if (e.key === 'ArrowDown') { e.preventDefault(); hi.value = Math.min(norm.value.length - 1, hi.value + 1); }
      else if (e.key === 'ArrowUp') { e.preventDefault(); hi.value = Math.max(0, hi.value - 1); }
      else if (e.key === 'Enter') { e.preventDefault(); if (hi.value >= 0) pick(norm.value[hi.value].value); }
      else if (e.key === 'Tab') { open.value = false; }
    };
    const onDoc = (e) => { if (root.value && !root.value.contains(e.target)) open.value = false; };
    onMounted(() => document.addEventListener('mousedown', onDoc));
    onBeforeUnmount(() => document.removeEventListener('mousedown', onDoc));
    return { open, hi, root, norm, label, pick, toggle, onKey };
  },
  template: `
  <div class="sf" :class="{ open, 'sf-sm': small }" ref="root">
    <button type="button" class="sf-btn" role="combobox" :disabled="disabled" :aria-expanded="open" aria-haspopup="listbox" @click="toggle" @keydown="onKey">
      <span style="overflow: hidden; text-overflow: ellipsis;">{{ label }}</span>
    </button>
    <div v-if="open" class="sf-menu" role="listbox">
      <button type="button" v-for="(o, i) in norm" :key="o.value" class="sf-opt" role="option"
        :class="{ sel: o.value === modelValue, hi: i === hi }" :aria-selected="o.value === modelValue"
        @click="pick(o.value)" @mousemove="hi = i" tabindex="-1">
        {{ o.label }}
        <span v-if="o.value === modelValue" class="tick">✓</span>
      </button>
    </div>
  </div>
  `,
};
