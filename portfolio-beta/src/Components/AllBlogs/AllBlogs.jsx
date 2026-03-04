import "./AllBlogs.css";
import notebook_data from "../../assets/notebook_data";
import markdown_data from "../../assets/markdown_data";
import paper_data from "../../assets/paper_data";
import { Link } from "react-router-dom";
import { useEffect } from "react";

const BlogSection = ({ title, items, renderLink }) => (
  <div className="blog-section">
    <h3 className="blog-section-title">{title}</h3>
    <div className="blog-section-grid">
      {items.map((item, index) => (
        <div key={index} className="blog-item">
          {renderLink(item)}
        </div>
      ))}
    </div>
  </div>
);

const AllBlogs = () => {
  useEffect(() => {
    window.scrollTo(0, 0);
  }, []);

  return (
    <div id="all-blogs" className="all-blogs">
      <div className="all-blogs-inner">
        <h1 className="all-blogs-title">Blog</h1>

        <BlogSection
          title="Research Paper Summary"
          items={paper_data}
          renderLink={(ppt) =>
            ppt.external === false ? (
              <Link className="blog-item-link" to={"/blogs/markdowns/" + ppt.link}>
                <h4>{ppt.name}</h4>
              </Link>
            ) : (
              <a
                href={ppt.link}
                target="_blank"
                rel="noopener noreferrer"
                className="blog-item-link"
              >
                <h4>{ppt.name}</h4>
                {ppt.contents?.length > 0 && (
                  <ul>
                    {ppt.contents.map((c, i) => (
                      <li key={i}>{c}</li>
                    ))}
                  </ul>
                )}
              </a>
            )
          }
        />

        <BlogSection
          title="Jupyter Notebooks"
          items={notebook_data}
          renderLink={(blog) => (
            <Link className="blog-item-link" to={"/blogs/notebooks/" + blog.link}>
              <h4>{blog.name}</h4>
              {blog.contents?.length > 0 && (
                <ul>
                  {blog.contents.map((c, i) => (
                    <li key={i}>{c}</li>
                  ))}
                </ul>
              )}
            </Link>
          )}
        />

        <BlogSection
          title="Markdown"
          items={markdown_data}
          renderLink={(md) => (
            <Link className="blog-item-link" to={"/blogs/markdowns/" + md.link}>
              <h4>{md.name}</h4>
              {md.contents?.length > 0 && (
                <ul>
                  {md.contents.map((c, i) => (
                    <li key={i}>{c}</li>
                  ))}
                </ul>
              )}
            </Link>
          )}
        />
      </div>
    </div>
  );
};

export default AllBlogs;
