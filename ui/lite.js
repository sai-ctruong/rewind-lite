"use strict";

(() => {
  const $ = (id) => document.getElementById(id);
  const el = {
    sidebar: $("sidebar"), scrim: $("menu-scrim"), menu: $("menu-toggle"),
    list: $("video-list"), listStatus: $("video-list-status"), videoCount: $("video-count"),
    upload: $("video-upload"), uploadStatus: $("upload-status"),
    player: $("video-player"), videoEmpty: $("video-empty"), videoTitle: $("video-title"),
    indexButton: $("index-button"), indexPanel: $("index-panel"), indexStatus: $("index-status"),
    searchForm: $("search-form"), query: $("search-query"), searchButton: $("search-button"),
    searchHelp: $("search-help"), results: $("results"), resultsStatus: $("results-status"),
    resultsCount: $("results-count"), track: $("moment-track"),
    timelineHint: $("timeline-hint"), timelineDuration: $("timeline-duration")
  };
  const state = {
    videos: [], selected: null, ready: false, busy: false, searching: false,
    results: [], activeFrame: null, menuOpen: false, needsRebuild: false, generation: 0
  };
  const phone = window.matchMedia("(max-width: 900px)");

  const messages = {
    unsupported_video: "Chỉ nhận video MP4 có thể phát.",
    upload_too_large: "Video vượt quá giới hạn 512 MB.",
    upload_failed: "Không lưu được video. Hãy thử lại.",
    index_missing: "Video chưa được lập chỉ mục.",
    index_incomplete: "Chỉ mục chưa hoàn chỉnh. Hãy lập lại.",
    index_expired: "Video hoặc cấu hình đã thay đổi. Hãy lập lại chỉ mục.",
    index_busy: "Đang xử lý video. Hãy chờ một chút.",
    index_failed: "Lập chỉ mục thất bại. Hãy thử lại.",
    model_oom: "Không đủ bộ nhớ để chạy mô hình.",
    search_failed: "Tìm kiếm thất bại. Hãy thử lại.",
    invalid_query: "Hãy nhập một câu mô tả ngắn.",
    video_not_found: "Không tìm thấy video đã chọn."
  };

  function message(error, fallback) {
    return messages[error?.code] || fallback;
  }

  async function api(path, options = {}) {
    let response;
    try {
      response = await fetch(`/api/lite${path}`, options);
    } catch {
      throw { code: "network", message: "network" };
    }
    let data;
    try {
      data = await response.json();
    } catch {
      throw { code: "bad_response", message: "bad response" };
    }
    if (!response.ok) throw data.error || { code: "request_failed" };
    return data;
  }

  function post(path, data) {
    return api(path, { method: "POST", headers: { "Content-Type": "application/json" },
                       body: JSON.stringify(data) });
  }

  function formatTime(value) {
    if (!Number.isFinite(value) || value < 0) return "—";
    const seconds = Math.floor(value);
    const minutes = Math.floor(seconds / 60);
    const suffix = String(seconds % 60).padStart(2, "0");
    if (minutes >= 60) return `${Math.floor(minutes / 60)}:${String(minutes % 60).padStart(2, "0")}:${suffix}`;
    return `${String(minutes).padStart(2, "0")}:${suffix}`;
  }

  function setMenu(open) {
    const wasOpen = state.menuOpen;
    state.menuOpen = open && phone.matches;
    document.body.classList.toggle("menu-open", state.menuOpen);
    el.menu.setAttribute("aria-expanded", String(state.menuOpen));
    el.menu.setAttribute("aria-label", state.menuOpen ? "Đóng danh sách video" : "Mở danh sách video");
    el.scrim.hidden = !state.menuOpen;
    el.sidebar.inert = phone.matches && !state.menuOpen;
    if (wasOpen && !state.menuOpen && phone.matches) el.menu.focus();
  }

  function controls() {
    el.indexButton.disabled = !state.selected || state.busy || state.searching;
    el.query.disabled = !state.ready || state.busy;
    el.searchButton.disabled = !state.ready || state.busy || state.searching;
    el.indexButton.textContent = state.needsRebuild ? "Lập lại chỉ mục" : "Lập chỉ mục";
    renderVideos();
  }

  function renderVideos() {
    el.list.replaceChildren();
    el.videoCount.textContent = String(state.videos.length);
    el.listStatus.hidden = state.videos.length > 0;
    for (const id of state.videos) {
      const li = document.createElement("li");
      const button = document.createElement("button");
      button.type = "button";
      button.disabled = state.busy || state.searching;
      button.setAttribute("aria-current", String(id === state.selected));
      button.setAttribute("title", `${id}.mp4`);
      const glyph = document.createElement("span");
      glyph.className = "video-glyph";
      glyph.setAttribute("aria-hidden", "true");
      const name = document.createElement("span");
      name.className = "video-name";
      name.textContent = `${id}.mp4`;
      button.append(glyph, name);
      button.addEventListener("click", () => selectVideo(id));
      li.append(button);
      el.list.append(li);
    }
  }

  function setIndexStatus(status, text) {
    el.indexPanel.dataset.state = status;
    el.indexStatus.textContent = text;
  }

  function setResultStatus(text, type = "") {
    el.resultsStatus.className = `results-status${type ? ` ${type}` : ""}`;
    el.resultsStatus.textContent = text;
    el.resultsStatus.hidden = false;
  }

  function clearResults(text = "Nhập mô tả để tìm khoảnh khắc trong video này.") {
    state.results = [];
    state.activeFrame = null;
    el.results.replaceChildren();
    el.resultsCount.textContent = "0";
    setResultStatus(text);
    renderTimeline();
  }

  function renderTimeline() {
    el.track.replaceChildren();
    const duration = el.player.duration;
    el.timelineDuration.textContent = formatTime(duration);
    if (!Number.isFinite(duration) || duration <= 0 || !state.results.length) {
      el.timelineHint.textContent = state.results.length ? "Đang đọc thời lượng video…" : "Các mốc sẽ xuất hiện sau khi tìm kiếm.";
      return;
    }
    el.timelineHint.textContent = "Chọn một mốc để tua video.";
    for (const hit of state.results) {
      const mark = document.createElement("button");
      mark.type = "button";
      mark.style.left = `${Math.min(98, Math.max(2, hit.timestamp / duration * 100))}%`;
      mark.className = hit.frame_id === state.activeFrame ? "is-active" : "";
      mark.setAttribute("aria-label", `Tua đến ${formatTime(hit.timestamp)}, kết quả hạng ${hit.rank}`);
      mark.title = `${formatTime(hit.timestamp)} · hạng ${hit.rank}`;
      mark.addEventListener("click", () => seekTo(hit));
      el.track.append(mark);
    }
  }

  function seekTo(hit) {
    const seek = () => {
      const duration = el.player.duration;
      el.player.currentTime = Number.isFinite(duration) ? Math.min(hit.timestamp, duration) : hit.timestamp;
      el.player.play().catch(() => {});
      state.activeFrame = hit.frame_id;
      el.timelineHint.textContent = `Đang ở mốc ${formatTime(hit.timestamp)}.`;
      for (const button of el.results.querySelectorAll("button")) {
        button.classList.toggle("is-active", button.dataset.frameId === hit.frame_id);
      }
      renderTimeline();
    };
    if (el.player.readyState >= 1) seek();
    else el.player.addEventListener("loadedmetadata", seek, { once: true });
  }

  function renderResults() {
    el.results.replaceChildren();
    el.resultsCount.textContent = String(state.results.length);
    if (!state.results.length) {
      setResultStatus("Không tìm thấy khung hình phù hợp. Hãy thử mô tả khác.");
      renderTimeline();
      return;
    }
    el.resultsStatus.hidden = true;
    for (const hit of state.results) {
      const li = document.createElement("li");
      const button = document.createElement("button");
      button.type = "button";
      button.className = "result-button";
      button.dataset.frameId = hit.frame_id;
      button.setAttribute("aria-label", `Kết quả hạng ${hit.rank}, ${formatTime(hit.timestamp)}, độ tương đồng ${hit.score.toFixed(3)}. Tua video tới mốc này.`);
      const image = document.createElement("img");
      image.className = "result-image";
      image.src = hit.image_url;
      image.loading = "lazy";
      image.alt = `Khung hình tại ${formatTime(hit.timestamp)}`;
      const copy = document.createElement("span");
      copy.className = "result-copy";
      const rank = document.createElement("span");
      rank.className = "result-rank";
      rank.textContent = `Hạng ${hit.rank}`;
      const time = document.createElement("span");
      time.className = "result-time";
      time.textContent = formatTime(hit.timestamp);
      const score = document.createElement("span");
      score.className = "result-score";
      score.textContent = `${hit.score_kind === "cosine" ? "Cosine" : hit.score_kind} · ${hit.score.toFixed(3)}`;
      copy.append(rank, time, score);
      button.append(image, copy);
      button.addEventListener("click", () => seekTo(hit));
      li.append(button);
      el.results.append(li);
    }
    renderTimeline();
  }

  async function pollProgress() {
    try {
      const data = await api("/progress");
      if (data.status === "indexing" && data.video_id === state.selected) {
        const elapsed = Number.isFinite(data.elapsed_s) ? `${data.elapsed_s.toFixed(1)} giây` : "đang xử lý";
        setIndexStatus("indexing", `Đã xử lý ${data.count} khung hình · ${elapsed}.`);
      }
    } catch { /* The indexing request reports its final error. */ }
  }

  async function indexVideo(rebuild = false) {
    if (!state.selected || state.busy) return;
    const id = state.selected;
    state.busy = true;
    state.ready = false;
    controls();
    el.searchHelp.textContent = "Đang lập chỉ mục video. Bạn có thể theo dõi trạng thái bên dưới.";
    setIndexStatus("indexing", "Đang chuẩn bị video và mô hình…");
    const timer = window.setInterval(pollProgress, 700);
    try {
      const result = await post("/index", { video_id: id, rebuild });
      if (id !== state.selected) return;
      state.ready = true;
      state.needsRebuild = false;
      if (result.poster_url && result.poster_url.startsWith(`/api/lite/frames/${encodeURIComponent(id)}/`)) {
        el.player.poster = result.poster_url;
      }
      setIndexStatus("ready", `${result.cached ? "Đã dùng lại chỉ mục" : "Lập chỉ mục xong"} · ${result.frames} khung hình có thể tìm kiếm.`);
      el.searchHelp.textContent = "Nhập mô tả rồi nhấn Enter hoặc Tìm kiếm.";
    } catch (error) {
      state.needsRebuild = error.code === "index_expired" || error.code === "index_incomplete";
      const detail = message(error, "Không thể lập chỉ mục video. Hãy thử lại.");
      setIndexStatus("error", detail);
      el.searchHelp.textContent = detail;
    } finally {
      window.clearInterval(timer);
      state.busy = false;
      controls();
    }
  }

  async function selectVideo(id) {
    if (state.busy || state.searching || !state.videos.includes(id)) return;
    state.generation += 1;
    state.selected = id;
    state.ready = false;
    state.needsRebuild = false;
    controls();
    clearResults();
    el.videoTitle.textContent = `${id}.mp4`;
    el.player.hidden = false;
    el.videoEmpty.hidden = true;
    el.player.removeAttribute("poster");
    el.player.src = `/api/lite/videos/${encodeURIComponent(id)}/stream`;
    el.player.load();
    setMenu(false);
    try {
      await post("/select", { video_id: id });
      await indexVideo();
    } catch (error) {
      setIndexStatus("error", message(error, "Không chọn được video. Hãy thử lại."));
      controls();
    }
  }

  async function loadVideos(preferred = null) {
    try {
      const data = await api("/videos");
      state.videos = Array.isArray(data.videos) ? data.videos : [];
      el.listStatus.textContent = state.videos.length ? "" : "Chưa có video. Tải một video MP4 lên để bắt đầu.";
      el.listStatus.classList.remove("error");
      renderVideos();
      const first = preferred && state.videos.includes(preferred) ? preferred :
                    data.selected_video && state.videos.includes(data.selected_video) ? data.selected_video : state.videos[0];
      if (first) await selectVideo(first);
      else setIndexStatus("idle", "Chưa có video. Hãy tải một video MP4 lên.");
    } catch {
      el.listStatus.textContent = "Không tải được danh sách video. Hãy tải lại trang.";
      el.listStatus.classList.add("error");
      setIndexStatus("error", "Không kết nối được với API.");
    }
  }

  async function uploadVideo(file) {
    if (!file || state.busy) return;
    if (!file.name.toLowerCase().endsWith(".mp4")) {
      el.uploadStatus.textContent = "Chỉ nhận video MP4.";
      el.uploadStatus.classList.add("error");
      return;
    }
    state.busy = true;
    controls();
    el.uploadStatus.textContent = "Đang tải video lên…";
    el.uploadStatus.classList.remove("error");
    try {
      const form = new FormData();
      form.append("video", file);
      const uploaded = await api("/upload", { method: "POST", body: form });
      el.uploadStatus.textContent = "Đã tải video lên.";
      state.busy = false;
      await loadVideos(uploaded.video_id);
    } catch (error) {
      el.uploadStatus.textContent = message(error, "Không tải được video. Hãy thử lại.");
      el.uploadStatus.classList.add("error");
    } finally {
      state.busy = false;
      el.upload.value = "";
      controls();
    }
  }

  async function search(event) {
    event.preventDefault();
    if (!state.ready || state.busy || state.searching) return;
    const query = el.query.value.trim();
    if (!query) {
      el.searchHelp.textContent = "Hãy nhập một câu mô tả để tìm kiếm.";
      el.searchHelp.classList.add("error");
      el.query.focus();
      return;
    }
    el.searchHelp.classList.remove("error");
    el.searchHelp.textContent = `Đang tìm “${query}”…`;
    state.searching = true;
    controls();
    clearResults("Đang tìm các khung hình phù hợp…");
    el.resultsStatus.classList.add("loading");
    const id = state.selected;
    const generation = state.generation;
    try {
      const data = await post("/search", { video_id: id, query, top_k: 5 });
      if (generation !== state.generation) return;
      state.results = (Array.isArray(data.results) ? data.results : []).filter((hit) =>
        Number.isFinite(hit.timestamp) && hit.timestamp >= 0 && Number.isFinite(hit.score) &&
        typeof hit.frame_id === "string" && typeof hit.image_url === "string" &&
        hit.image_url.startsWith(`/api/lite/frames/${encodeURIComponent(id)}/`)).slice(0, 5);
      renderResults();
      el.searchHelp.textContent = state.results.length ? `Đã tìm thấy ${state.results.length} khung hình.` : "Thử mô tả ngắn hơn hoặc dùng từ khác.";
    } catch (error) {
      setResultStatus(message(error, "Tìm kiếm thất bại. Hãy thử lại."), "error");
      el.searchHelp.textContent = "Bạn có thể thử tìm lại.";
    } finally {
      state.searching = false;
      controls();
    }
  }

  el.menu.addEventListener("click", () => setMenu(!state.menuOpen));
  el.scrim.addEventListener("click", () => setMenu(false));
  document.addEventListener("keydown", (event) => { if (event.key === "Escape") setMenu(false); });
  phone.addEventListener("change", () => setMenu(false));
  el.sidebar.inert = phone.matches;
  el.upload.addEventListener("change", () => uploadVideo(el.upload.files[0]));
  el.indexButton.addEventListener("click", () => indexVideo(state.needsRebuild));
  el.searchForm.addEventListener("submit", search);
  el.player.addEventListener("loadedmetadata", renderTimeline);
  loadVideos();
})();
