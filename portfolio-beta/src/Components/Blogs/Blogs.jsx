import "./Blogs.css";
import notebook_data from "../../assets/notebook_data";
import markdown_data from "../../assets/markdown_data";
import paper_data from "../../assets/paper_data";
import { Link } from "react-router-dom";

const BlogCard = ({ item, href, isExternal, showContents = true }) => {
  const content = (
    <>
      <h3>{item.name}</h3>
      {showContents && item.contents && item.contents.length > 0 && (
        <ul>
          {item.contents.slice(0, 3).map((c, i) => (
            <li key={i}>{c}</li>
          ))}
        </ul>
      )}
    </>
  );

  if (isExternal) {
    return (
      <a className="blog-card" href={href} target="_blank" rel="noopener noreferrer">
        {content}
      </a>
    );
  }
  return (
    <Link className="blog-card" to={href}>
      {content}
    </Link>
  );
};

const Blogs = () => {
  const paperItems = paper_data.slice(0, 2);
  const notebookItems = notebook_data.slice(0, 2);
  const markdownItems = markdown_data.slice(0, 2);

  return (
    <section id="blog" className="blog">
      <div className="section-inner">
        <h2 className="section-title">Blog</h2>

        {paperItems.length > 0 && (
          <div className="blog-subsection">
            <h3 className="blog-subsection-title">Research Paper Summary</h3>
            <div className="blog-grid">
              {paperItems.map((item, index) => (
                <BlogCard
                  key={index}
                  item={item}
                  href={item.external ? item.link : "/blogs/markdowns/" + item.link}
                  isExternal={item.external !== false}
                  showContents={false}
                />
              ))}
            </div>
          </div>
        )}

        {notebookItems.length > 0 && (
          <div className="blog-subsection">
            <h3 className="blog-subsection-title">Jupyter Notebooks</h3>
            <div className="blog-grid">
              {notebookItems.map((item, index) => (
                <BlogCard
                  key={index}
                  item={item}
                  href={"/blogs/notebooks/" + item.link}
                  isExternal={false}
                />
              ))}
            </div>
          </div>
        )}

        {markdownItems.length > 0 && (
          <div className="blog-subsection">
            <h3 className="blog-subsection-title">Markdown</h3>
            <div className="blog-grid">
              {markdownItems.map((item, index) => (
                <BlogCard
                  key={index}
                  item={item}
                  href={"/blogs/markdowns/" + item.link}
                  isExternal={false}
                />
              ))}
            </div>
          </div>
        )}

        <Link className="blog-view-all" to="/blogs">
          View all blogs →
        </Link>
      </div>
    </section>
  );
};

export default Blogs;
