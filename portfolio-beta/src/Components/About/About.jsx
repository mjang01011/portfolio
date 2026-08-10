import "./About.css";
import { Fragment, useState } from "react";
import { Link } from "react-router-dom";
import sail_logo from "../../assets/sail_logo.jpg";
import duke_logo from "../../assets/duke_logo.png";
import nvidia_logo from "../../assets/nvidia_logo.png";
import google_logo from "../../assets/google_logo.png";
import amd_logo from "../../assets/amd_logo.png";
import gpic_thumbnail from "../../assets/gpic.jpeg";

const orgLogos = {
  "Stanford Vision and Learning Lab (SVL)": sail_logo,
  "Advised by Dr. Lawrence Carin, Duke University": duke_logo,
  "aiMIND Study Group, Duke University": duke_logo,
  NVIDIA: nvidia_logo,
  Google: google_logo,
  "AMD (Advanced Micro Devices, Inc.)": amd_logo,
  "Duke University": duke_logo,
};

const experiences = [
  {
    id: "svl-researcher",
    group: "research",
    title: "Machine Learning Researcher",
    location: "Stanford, CA, USA",
    org: "Stanford Vision and Learning Lab (SVL)",
    dates: "Jan 2026 – Current",
    incoming: false,
    details: [
      "Developing new visual generation dataset and benchmarking state-of-the-art image generation models.",
    ],
  },
  {
    id: "carin-researcher",
    group: "research",
    title: "Machine Learning Researcher",
    location: "Durham, NC, USA",
    org: "Advised by Dr. Lawrence Carin, Duke University",
    dates: "Jan 2025 – May 2025",
    incoming: false,
    details: [
      "Developed a scalable Bag-of-Words topic model that learns interpretable low-dimensional document embeddings.",
      "Extended the model to integrate document-level question-answer data, designing a double-softmax latent factor model to identify narratives that explain structured QA responses across documents.",
    ],
  },
  {
    id: "aimind-researcher",
    group: "research",
    title: "Machine Learning Researcher",
    location: "Durham, NC, USA",
    org: "aiMIND Study Group, Duke University",
    dates: "Mar 2024 – May 2025",
    incoming: false,
    details: [
      "Led the development of a multimodal deep learning model in PyTorch to predict neurodegenerative disease progression, and built scalable pipelines to process 6,000+ retinal images for end-to-end training.",
      "Achieved AUC of 0.985 on Mild Cognitive Impairment prediction with a fused multimodal EfficientNet-B6.",
    ],
  },
  {
    id: "nvidia-intern",
    group: "experience",
    title: "Incoming Software Engineering Intern",
    location: "",
    org: "NVIDIA",
    dates: "Incoming Fall 2026",
    incoming: true,
    details: ["GPU Compute Software QA"],
  },
  {
    id: "google-intern",
    group: "experience",
    title: "Software Engineering Intern",
    location: "",
    org: "Google",
    dates: "June 2026 – September 2026",
    incoming: false,
    details: [
      "Engineered a Gemini multi-agent postmortem generation pipeline for the Google Cloud Networking team to automate data collection, incident analysis, and reduce report drafting time from days to hours.",
      "Implemented a scalable evaluation framework utilizing an LLM-as-a-Judge architecture and human-in-the-loop refinement to guarantee the technical accuracy and reliability of generated incident diagnostics.",
    ],
  },
  {
    id: "amd-intern",
    group: "experience",
    title: "Software Engineering Intern",
    location: "Austin, TX, USA",
    org: "AMD (Advanced Micro Devices, Inc.)",
    dates: "June 2025 – September 2025",
    incoming: false,
    details: [
      "Designed an agentic system using FastMCP and RAG to interface a production-ready LLM-powered assistant with internal triage tools, accelerating silicon log analysis and automated failure triage for next-generation server CPUs.",
      "Spearheaded the development of a LLM-powered debugging assistant that translates engineers' natural language requests into optimized MongoDB queries for efficient data retrieval and diagnostics.",
      "Established a self-hosted MongoDB sync pipeline from Cosmos DB using Python and GitHub Actions, achieving faster access latency, reduced API costs, and elimination of rate limiter failures through read/write optimization.",
    ],
  },
  {
    id: "duke-ta",
    group: "experience",
    title: "Teaching Assistant",
    location: "Durham, NC, USA",
    org: "Duke University",
    dates: "2024 Fall – 2025 Spring",
    incoming: false,
    details: [
      "2025 Spring: ECE580 Introduction to Machine Learning (Graduate). Key concepts include discriminative/generative classifiers (SVM, RVM, logistic regression, Bayes), dimensionality reduction, feature selection, and performance evaluation.",
      "2024 Fall: ECE480 Applied Probability for Statistical Learning. Key concepts include Bayesian inference, probabilistic reasoning, mixture models, and model selection.",
      "2024 Fall: CS371 Elements of Machine Learning. Key concepts include linear/nonlinear SVM, kernels, decision trees, CNNs, and transformers.",
    ],
  },
];

const publications = [
  {
    title: "GPIC: A Giant Permissive Image Corpus for Visual Generation",
    thumbnail: gpic_thumbnail,
    authors: [
      { name: "Keshigeyan Chandrasegaran", marks: "*,1" },
      { name: "Kyle Sargent", marks: "*,1" },
      { name: "Suchir Agarwal", marks: "1" },
      { name: "Michael Jang", marks: "1", highlight: true },
      { name: "Michael Poli", marks: "1,2" },
      { name: "Juan Carlos Niebles", marks: "1,4" },
      { name: "Justin Johnson", marks: "3" },
      { name: "Jiajun Wu", marks: "1" },
      { name: "Li Fei-Fei", marks: "1" },
    ],
    affiliations: [
      { mark: "1", name: "Stanford University" },
      { mark: "2", name: "Radical Numerics" },
      { mark: "3", name: "University of Michigan" },
      { mark: "4", name: "Salesforce Research" },
    ],
    note: "* Equal contribution",
    links: [
      { label: "Project Page", href: "https://gpic.stanford.edu/" },
      { label: "Paper", href: "https://arxiv.org/abs/2605.30341" },
      { label: "Dataset", href: "https://huggingface.co/datasets/stanford-vision-lab/gpic" },
      { label: "Code", href: "https://github.com/keshik6/gpic" },
    ],
  },
];

const PublicationCard = ({ publication }) => {
  return (
    <article className="pub-card">
      <Link
        className="pub-thumb"
        to={publication.links[0].href}
        target="_blank"
        rel="noopener noreferrer"
        aria-label={`${publication.title} project page`}
      >
        <img src={publication.thumbnail} alt={publication.title} />
      </Link>
      <div className="pub-body">
        <h4 className="pub-title">
          <Link to={publication.links[0].href} target="_blank" rel="noopener noreferrer">
            {publication.title}
          </Link>
        </h4>
        <p className="pub-authors">
          {publication.authors.map((author, i) => (
            <Fragment key={author.name}>
              <span className="pub-author">
                <span className={author.highlight ? "pub-author-me" : undefined}>{author.name}</span>
                <sup>{author.marks}</sup>
              </span>
              {i < publication.authors.length - 1 && ", "}
            </Fragment>
          ))}
        </p>
        <p className="pub-affiliations">
          {publication.affiliations.map((affiliation) => (
            <span key={affiliation.mark}>
              <sup>{affiliation.mark}</sup>
              {affiliation.name}
            </span>
          ))}
          <span className="pub-note">{publication.note}</span>
        </p>
        <div className="pub-links">
          {publication.links.map((link) => (
            <Link
              key={link.label}
              className="pub-link"
              to={link.href}
              target="_blank"
              rel="noopener noreferrer"
            >
              {link.label}
            </Link>
          ))}
        </div>
      </div>
    </article>
  );
};

const ExperienceCard = ({ experience, isExpanded, onToggle }) => {
  return (
    <div
      className={`timeline-card ${isExpanded ? "expanded" : ""} ${experience.incoming ? "incoming" : ""}`}
      onClick={onToggle}
    >
      <div className="timeline-card-header">
        <div className="timeline-card-header-text">
          <h4 className="timeline-title">{experience.title}</h4>
          <p className="timeline-org">{experience.org}</p>
          <p className="timeline-dates">{experience.dates}</p>
        </div>
        <div className="timeline-header-right">
          {orgLogos[experience.org] && (
            <img src={orgLogos[experience.org]} alt={experience.org} className="timeline-org-logo" />
          )}
          <span className="timeline-expand-icon">{isExpanded ? "−" : "+"}</span>
        </div>
      </div>
      <div className={`timeline-details-wrapper ${isExpanded ? "expanded" : ""}`}>
        <ul className="timeline-details">
          {experience.details.map((detail, i) => (
            <li key={i}>{detail}</li>
          ))}
        </ul>
      </div>
    </div>
  );
};

const About = () => {
  const [expandedId, setExpandedId] = useState(null);

  const researchExperiences = experiences.filter((exp) => exp.group === "research");
  const experiencesList = experiences.filter((exp) => exp.group === "experience");

  const toggleCard = (id) => (e) => {
    e.stopPropagation();
    setExpandedId((current) => (current === id ? null : id));
  };

  return (
    <section id="about" className="about">
      <div className="section-inner">
        <h2 className="section-title">About Me</h2>
        <div className="about-intro">
          <p>
            I am passionate about <strong>multimodal large language models</strong>: models that understand and generate across text, images, and other modalities.
          </p>
          <p>
            I thrive in collaborative research environments and enjoy integrating new tools and methods into my work. Outside of research, I love travel photography and astrophotography.
          </p>
        </div>

        <div className="timeline-wrapper">
          <div className="timeline-section" id="experiences">
            <h3 className="timeline-section-title">Experiences</h3>
            <div className="timeline-scroll">
              {experiencesList.map((exp) => (
                <ExperienceCard
                  key={exp.id}
                  experience={exp}
                  isExpanded={expandedId === exp.id}
                  onToggle={toggleCard(exp.id)}
                />
              ))}
            </div>
          </div>
          <div className="timeline-section" id="research">
            <h3 className="timeline-section-title">Research</h3>
            <div className="timeline-scroll">
              {researchExperiences.map((exp) => (
                <ExperienceCard
                  key={exp.id}
                  experience={exp}
                  isExpanded={expandedId === exp.id}
                  onToggle={toggleCard(exp.id)}
                />
              ))}
            </div>
          </div>
          <div className="timeline-section" id="publications">
            <h3 className="timeline-section-title">Publications</h3>
            <div className="pub-list">
              {publications.map((publication) => (
                <PublicationCard key={publication.title} publication={publication} />
              ))}
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default About;
